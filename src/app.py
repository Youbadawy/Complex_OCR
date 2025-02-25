import os
import re
import json
import logging
import platform
import streamlit as st
import pdfplumber
import pandas as pd
import numpy as np
from PIL import Image
import pytesseract
import cv2
import hashlib
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import concurrent.futures
import sqlite3
from pathlib import Path
from nltk.corpus import stopwords
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from transformers import pipeline, DonutProcessor, VisionEncoderDecoderModel
import torch
from huggingface_hub import login
from groq import Groq
import ocr_utils
import chatbot_utils

# Constants
DB_PATH = "mammo_reports.db"
BATCH_SIZE = 10

def process_pdf(uploaded_file):
    """Process PDF as a complete document with combined text from all pages"""
    try:
        # Generate file hash for caching
        file_hash = hashlib.sha256(uploaded_file.getvalue()).hexdigest()
        
        # Check cache
        if Path(DB_PATH).exists():
            with sqlite3.connect(DB_PATH) as conn:
                cached = conn.execute("""
                    SELECT findings, metadata FROM reports
                    WHERE md5_hash = ?
                """, (file_hash,)).fetchone()
                if cached:
                    return json.loads(cached[0]), json.loads(cached[1])
        
        # Extract text from all pages
        all_text = []
        with pdfplumber.open(uploaded_file) as pdf:
            # Process all pages and combine text
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    all_text.append(page_text)
                else:
                    # Fallback to OCR if text extraction fails
                    img = page.to_image(resolution=300).original
                    ocr_text = ocr_utils.simple_ocr(np.array(img))
                    all_text.append(ocr_text)
        
        # Combine all text into a single document
        combined_text = "\n\n".join(all_text)
        
        # Process the combined text to extract all fields
        structured_data = ocr_utils.process_document_text(combined_text)
        
        # Add metadata
        metadata = {
            "md5_hash": file_hash,
            "filename": uploaded_file.name,
            "pages": len(all_text),
            "processing_date": pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Save to database
        save_to_db({
            "md5_hash": file_hash,
            "filename": uploaded_file.name,
            "patient_name": structured_data.get('patient_name', 'Unknown'),
            "exam_date": structured_data.get('exam_date', 'Unknown'),
            "findings": json.dumps(structured_data),
            "metadata": json.dumps(metadata)
        })
        
        return structured_data, metadata
        
    except Exception as e:
        logging.error(f"PDF processing error: {str(e)}", exc_info=True)
        return ocr_utils.default_structured_output(), {"error": str(e), "filename": uploaded_file.name}

# Streamlit UI
st.title("Medical Report Processor")

uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

if uploaded_files:
    all_results = []
    
    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Process each PDF as a complete document
    for i, file in enumerate(uploaded_files):
        status_text.text(f"Processing {file.name}...")
        
        # Process the PDF
        structured_data, metadata = process_pdf(file)
        
        # Add to results
        all_results.append(structured_data)
        
        # Update progress
        progress = (i + 1) / len(uploaded_files)
        progress_bar.progress(progress)
    
    # Create DataFrame with one row per PDF
    if all_results:
        # Ensure consistent columns
        required_columns = [
            'birads_score', 'document_date', 'document_type', 
            'electronically_signed_by', 'exam_date', 'impression_result', 
            'mammogram_results', 'patient_history', 'patient_name', 
            'recommendation', 'testing_provider', 'ultrasound_results'
        ]
        
        # Create DataFrame with consistent columns
        df = pd.DataFrame(all_results)
        
        # Add missing columns with default values
        for col in required_columns:
            if col not in df.columns:
                df[col] = "Unknown"
        
        # Store in session state
        st.session_state.df = df
        
        # Display results
        st.success(f"Processed {len(df)} documents")
        st.dataframe(df)
        
        # Download button
        st.download_button(
            "Download Results",
            df.to_csv(index=False),
            "medical_data.csv",
            "text/csv"
        )

# Define common medical stopwords to exclude
STOP_WORDS = set(stopwords.words('english')).union({
    'patient', 'exam', 'result', 'findings', 'breast', 
    'mammogram', 'birads', 'assessment', 'clinical'
})

# Import OCR functions from utilities
import ocr_utils

# Handle Hugging Face authentication
import dotenv
dotenv.load_dotenv()  # Load .env file first
HF_TOKEN = os.getenv("HF_API_KEY")
if not HF_TOKEN:
    st.error(
        "Hugging Face API key required\n\n"
        "1. Get your token at https://huggingface.co/settings/tokens\n"
        "2. Create a .env file in project root with:\n"
        "   HF_API_KEY=your_token_here\n"
        "3. Restart the app"
    )
    st.stop()  # Prevent further execution
else:
    try:
        login(token=HF_TOKEN, add_to_git_credential=False)
    except Exception as e:
        st.error(f"Authentication failed: {str(e)}")
        st.stop()

# Initialize PaddleOCR models
@st.cache_resource(show_spinner="Initializing OCR Engine...")
def init_paddle():
    """Initialize PaddleOCR with model verification"""
    from paddleocr import PaddleOCR
    import paddle
    
    try:
        # Clear existing CUDA cache
        paddle.disable_static()
        if paddle.is_compiled_with_cuda():
            paddle.device.cuda.empty_cache()
        
        # Initialize with verified settings
        ocr = PaddleOCR(
            use_angle_cls=True,
            lang='en',
            use_gpu=False,
            enable_mkldnn=True,
            det_model_dir=os.path.expanduser('~/.paddleocr/whl/det/en/en_PP-OCRv3_det_infer'),
            rec_model_dir=os.path.expanduser('~/.paddleocr/whl/rec/en/en_PP-OCRv3_rec_infer'),
            cls_model_dir=os.path.expanduser('~/.paddleocr/whl/cls/ch_ppocr_mobile_v2.0_cls_infer')
        )
        
        # Validate with test OCR
        test_img = np.zeros((300, 300, 3), dtype=np.uint8)
        cv2.putText(test_img, "BIRADS 2", (50,150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        result = ocr.ocr(test_img, cls=True)
        
        if not result or "BIRADS" not in str(result):
            raise RuntimeError("PaddleOCR validation failed")
            
        return ocr
        
    except Exception as e:
        logging.critical(f"PaddleOCR init failed: {str(e)}")
        st.error(f"OCR Engine failed: {str(e)} - Check model files in ~/.paddleocr")
        return None

# Add this after imports but before OCR processing
if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
else:
    pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'  # Linux/Mac path

# Configure logging
logging.basicConfig(
    filename=os.path.join(os.path.dirname(__file__), 'mammo_ai.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    force=True
)

# Database functions
def init_db():
    """Initialize SQLite database with schema"""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS reports (
                md5_hash TEXT PRIMARY KEY,
                filename TEXT,
                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                patient_name TEXT,
                exam_date TEXT,
                findings TEXT,
                metadata TEXT
            )
        """)

# Initialize Donut model
@st.cache_resource
def init_donut():
    """Initialize Donut model with enhanced validation"""
    try:
        processor = DonutProcessor.from_pretrained(
            "naver-clova-ix/donut-base-finetuned-docvqa",
            revision="official"  # Pin to stable version
        )
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            torch.cuda.empty_cache()
            
        model = VisionEncoderDeco # Ericsson/codechecker
# -------------------------------------------------------------------------
#
#  Part of the CodeChecker project, under the Apache License v2.0 with
#  LLVM Exceptions. See LICENSE for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# -------------------------------------------------------------------------
"""
Defines the CodeChecker action for parsing a set of analysis results into a
human-readable format.
"""

import argparse
import os
import sys

from typing import Dict, List, Optional, Set, Tuple

from codechecker_report_converter.report import report_file, \
    reports as reports_helper
from codechecker_report_converter.report.output import baseline, codeclimate, \
    gerrit, json as report_to_json, plaintext
from codechecker_report_converter.report.output.html import \
    html as report_to_html
from codechecker_report_converter.report.statistics import Statistics

from codechecker_common import arg, logger
from codechecker_common.skiplist_handler import SkipListHandler
from codechecker_common.source_code_comment_handler import \
    REVIEW_STATUS_VALUES, SourceCodeCommentHandler
from codechecker_common.util import load_json

LOG = logger.get_logger('system')


def get_argparser_ctor_args():
    """
    This method returns a dict containing the kwargs for constructing an
    argparse.ArgumentParser (either directly or as a subparser).
    """

    return {
        'prog': 'CodeChecker parse',
        'formatter_class': arg.RawDescriptionDefaultHelpFormatter,

        # Description is shown when the command's help is queried directly
        'description': """
Parse and pretty-print the summary and results from one or more
'codechecker-analyze' result files. Bugs which are commented by using
"false_positive", "suppress" and "intentional" source code comments will not be
printed by the `parse` command.""",

        # Help is shown when the "parent" CodeChecker command lists the
        # individual subcommands.
        'help': "Print analysis summary and results in a human-readable format."
    }


def add_arguments_to_parser(parser):
    """
    Add the subcommand's arguments to the given argparse.ArgumentParser.
    """

    parser.add_argument('input',
                        type=str,
                        nargs='+',
                        metavar='file/folder',
                        help="The analysis result files and/or folders "
                             "containing analysis results which should be "
                             "parsed and printed.")

    parser.add_argument('-t', '--type', '--input-format',
                        dest="input_format",
                        required=False,
                        choices=['plist'],
                        default='plist',
                        help="Specify the format the analysis results were "
                             "created as.")

    output_opts = parser.add_argument_group("export arguments")
    output_opts.add_argument('-e', '--export',
                             dest="export",
                             required=False,
                             choices=['html', 'json', 'codeclimate', 'gerrit',
                                      'baseline'],
                             help="Specify extra output format type.")

    output_opts.add_argument('-o', '--output',
                             dest="output_path",
                             default=argparse.SUPPRESS,
                             help="Store the output in the given folder.")

    output_opts.add_argument('--url',
                             type=str,
                             dest="trim_path_prefix",
                             default=argparse.SUPPRESS,
                             help="Path prefix to trim when generating "
                                  "gerrit output.")

    parser.add_argument('--suppress',
                        type=str,
                        dest="suppress",
                        default=argparse.SUPPRESS,
                        required=False,
                        help="Path of the suppress file to use. Records in the "
                             "suppress file are used to suppress the "
                             "display of certain results when parsing the "
                             "analyses' report. (Reports to an analysis "
                             "result can also be suppressed in the source "
                             "code -- please consult the manual on how to "
                             "do so.) NOTE: The suppress file relies on the "
                             "\"bug identifier\" generated by the analyzers "
                             "which is experimental, take care when relying "
                             "on it.")

    parser.add_argument('--export-source-suppress',
                        dest="create_suppress",
                        action="store_true",
                        required=False,
                        default=argparse.SUPPRESS,
                        help="Write suppress data from the suppression "
                             "annotations found in the source files that were "
                             "analyzed earlier that created the results. "
                             "The suppression information will be written "
                             "to the parameter of '--suppress'.")

    parser.add_argument('--print-steps',
                        dest="print_steps",
                        action="store_true",
                        required=False,
                        default=argparse.SUPPRESS,
                        help="Print the steps the analyzers took in finding "
                             "the reported defect.")

    parser.add_argument('--review-status',
                        nargs='*',
                        dest="review_status",
                        metavar='REVIEW_STATUS',
                        choices=REVIEW_STATUS_VALUES,
                        default=["confirmed", "unreviewed"],
                        help="Filter results by review statuses. Valid "
                             f"values are: {', '.join(REVIEW_STATUS_VALUES)}.")

    parser.add_argument('--trim-path-prefix',
                        type=str,
                        nargs='*',
                        dest="trim_path_prefix",
                        required=False,
                        default=argparse.SUPPRESS,
                        help="Removes leading path from files which will be "
                             "printed. For instance if you analyze files "
                             "'/home/jsmith/my_proj/x.cpp' and "
                             "'/home/jsmith/my_proj/y.cpp', but would prefer "
                             "to see just 'x.cpp' and 'y.cpp' in the output, "
                             "this flag can help you. "
                             "Use '--trim-path-prefix=/home/jsmith/my_proj/' "
                             "to remove the leading path.")

    parser.add_argument('--verbose',
                        type=str,
                        dest="verbosity",
                        choices=['info', 'debug', 'debug_analyzer'],
                        default=argparse.SUPPRESS,
                        help="Set verbosity level.")

    parser.add_argument('--skip',
                        type=str,
                        dest="skipfile",
                        default=argparse.SUPPRESS,
                        help="Path to the Skipfile dictating which project "
                             "files should be omitted from analysis. Please "
                             "consult the User guide on how a Skipfile "
                             "should be laid out.")

    parser.add_argument('--config',
                        dest='config_file',
                        required=False,
                        help="R|Allow the configuration from an explicit JSON "
                             "based configuration file. The values configured "
                             "in the config file will overwrite the values "
                             "set in the command line. The format of "
                             "configuration file is:\n"
                             "{\n"
                             "  \"analyze\": [\n"
                             "    \"--enable=core.DivideZero\",\n"
                             "    \"--enable=core.CallAndMessage\",\n"
                             "    \"--report-hash=context-free-v2\",\n"
                             "    \"--verbose=debug\",\n"
                             "    \"--skip=$HOME/project/skip.txt\",\n"
                             "    \"--clean\"\n"
                             "  ]\n"
                             "}")

    logger.add_verbose_arguments(parser)
    parser.set_defaults(func=main)


def parse(plist_file: str, metadata_dict: Dict, skip_handler: SkipListHandler,
          trim_path_prefixes: Optional[List[str]] = None) -> \
        Tuple[List, List, Dict, Dict, Dict, Dict, Dict, Dict]:
    """
    Parses a plist report file.
    Returns:
        - list of source files
        - list of reports
        - map of source files to reports
        - report statistics
        - map of source files to analyzer statistics
        - map of source files to analyzer checkers
        - map of source files to analyzer runtimes
        - map of source files to analyzer failed checkers
    """
    files = []
    reports = []
    file_sources_map = {}
    file_stats = {}
    file_checkers_map = {}
    file_runtime_map = {}
    file_failed_checkers_map = {}

    source_code_comment_handler = SourceCodeCommentHandler()

    all_reports, metadata_dict, analyzer_statistics = \
        report_file.get_report_data(plist_file, metadata_dict)

    file_stats.update(analyzer_statistics)

    for report in all_reports:
        path = report.file_path

        if path:
            if skip_handler and skip_handler.should_skip(path):
                continue

            if trim_path_prefixes:
                for prefix in trim_path_prefixes:
                    if prefix and path.startswith(prefix):
                        path = path[len(prefix):]
                        break

            report.trim_path_prefixes(trim_path_prefixes)

            # Skip the report if it is a deduplication.
            if report.main:
                if path not in file_sources_map:
                    try:
                        with open(report.file_path,
                                  encoding='utf-8',
                                  errors='ignore') as source_file:
                            source = source_file.read()
                    except (OSError, IOError):
                        source = ""
                    file_sources_map[path] = source

                if source_code_comment_handler.has_source_line_comments(
                        report.file_path, report.line, file_sources_map):
                    continue

                report.review_status = \
                    source_code_comment_handler.get_suppressed(
                        report.file_path,
                        report.line,
                        report.checker_name,
                        file_sources_map)

                if report.review_status == 'false_positive':
                    report.review_status = 'false positive'

                if report.review_status == 'intentional':
                    report.review_status = 'intentional'

                reports.append(report)
                files.append(path)

    # Get report statistics.
    statistics = Statistics()
    statistics.num_of_analyzer_result_files = 1
    statistics.num_of_reports = len(reports)
    statistics.num_of_reports_with_source = len(reports)

    for report in reports:
        statistics.add_report(report)

    # Get analyzer statistics.
    statistics.num_of_analyzer_result_files = 1
    for analyzer_type, res in metadata_dict.items():
        analyzer_statistics = {}
        analyzer_statistics['version'] = res.get('analyzer_statistics', {})
        analyzer_statistics['failed'] = res.get('failed', {})
        analyzer_statistics['successful'] = res.get('successful', {})
        analyzer_statistics['runtime'] = res.get('runtime', {})

        checkers = res.get('checkers', {})
        analyzer_statistics['checkers'] = checkers

        if metadata_dict and analyzer_type in metadata_dict and \
                'analyzer_statistics' in metadata_dict[analyzer_type]:
            metadata = metadata_dict[analyzer_type]
            analyzer_statistics['analyzer_statistics'] = \
                metadata.get('analyzer_statistics', {})

            for curr_file, failed_checkers in metadata.get('failed', {}).items():
                if curr_file not in file_failed_checkers_map:
                    file_failed_checkers_map[curr_file] = {}

                file_failed_checkers_map[curr_file][analyzer_type] = \
                    failed_checkers

            for curr_file, checker_names in metadata.get('checkers', {}).items():
                if curr_file not in file_checkers_map:
                    file_checkers_map[curr_file] = {}

                file_checkers_map[curr_file][analyzer_type] = checker_names

            for curr_file, runtime in metadata.get('runtime', {}).items():
                if curr_file not in file_runtime_map:
                    file_runtime_map[curr_file] = {}

                file_runtime_map[curr_file][analyzer_type] = runtime

    return (
        files,
        reports,
        file_sources_map,
        statistics.checker_statistics,
        file_stats,
        file_checkers_map,
        file_runtime_map,
        file_failed_checkers_map
    )


def main(args):
    """
    Entry point for parsing some analysis results and printing them to the
    stdout in a human-readable format.
    """
    logger.setup_logger(args.verbose if 'verbose' in args else None)

    # Load configuration file if given.
    if 'config_file' in args:
        cfg = load_json(args.config_file, default={})
        if 'parse' in cfg:
            for k, v in cfg['parse'].items():
                # `args` object has priority over config file options.
                if k not in args:
                    setattr(args, k, v)

    export = args.export if 'export' in args else None
    if export == 'html' and 'output_path' not in args:
        LOG.error("Argument --output is required if HTML output is used.")
        sys.exit(1)

    if export == 'gerrit' and 'trim_path_prefix' not in args:
        LOG.error("Argument --url is required if gerrit output is used.")
        sys.exit(1)

    if export and export not in ['baseline', 'html', 'gerrit']:
        output_path = args.output_path if 'output_path' in args else None
        if output_path is None:
            LOG.error("Argument --output is required if JSON output is used.")
            sys.exit(1)

    if 'trim_path_prefix' in args:
        args.trim_path_prefix = \
            [prefix.rstrip('/') for prefix in args.trim_path_prefix]
    else:
        args.trim_path_prefix = None

    if 'suppress' in args:
        if not os.path.isfile(args.suppress):
            LOG.error("Suppress file '%s' given, but it does not exist",
                      args.suppress)
            sys.exit(1)

    skip_handler = None
    if 'skipfile' in args:
        try:
            with open(args.skipfile, encoding='utf-8', errors='ignore') as f:
                skip_handler = SkipListHandler(f.read())
        except (IOError, OSError) as err:
            LOG.error("Failed to open skip file: %s", err)
            sys.exit(1)

    input_files = set()
    for input_path in args.input:
        input_path = os.path.abspath(input_path)
        if os.path.isfile(input_path):
            input_files.add(input_path)
        elif os.path.isdir(input_path):
            input_paths = []
            try:
                input_paths = [os.path.join(input_path, filename)
                               for filename in os.listdir(input_path)]
            except OSError as err:
                LOG.error("Failed to get files from directory %s: %s",
                          input_path, err)
                sys.exit(1)

            input_files.update(input_paths)

    if not input_files:
        LOG.error("No input file was given.")
        sys.exit(1)

    all_reports = []
    statistics = Statistics()
    file_sources_map = {}
    file_stats = {}
    file_checkers_map = {}
    file_runtime_map = {}
    file_failed_checkers_map = {}

    metadata_dict = {}
    for input_file in input_files:
        if not os.path.exists(input_file):
            LOG.warning("Input file does not exist: %s", input_file)
            continue

        if os.path.isdir(input_file):
            LOG.warning("Input path is a directory: %s", input_file)
            continue

        LOG.debug("Parsing input file '%s'", input_file)

        if args.input_format == 'plist':
            try:
                files, reports, sources_map, checker_statistics, stats, \
                    checkers_map, runtime_map, failed_checkers_map = \
                    parse(input_file, metadata_dict, skip_handler,
                          args.trim_path_prefix)

                all_reports.extend(reports)
                statistics.num_of_analyzer_result_files += 1
                statistics.num_of_reports += len(reports)
                statistics.num_of_reports_with_source += len(reports)

                for checker_name, res in checker_statistics.items():
                    statistics.add_checker_statistics(checker_name, res)

                file_sources_map.update(sources_map)
                file_stats.update(stats)
                file_checkers_map.update(checkers_map)
                file_runtime_map.update(runtime_map)
                file_failed_checkers_map.update(failed_checkers_map)
            except Exception as ex:
                LOG.error("Parsing the plist failed: %s", str(ex))
        else:
            LOG.error("Unsupported input format: %s", args.input_format)
            sys.exit(1)

    if 'review_status' in args:
        all_reports = [r for r in all_reports
                       if r.review_status in args.review_status]

    all_reports = sorted(all_reports, key=lambda r: r.file_path)

    # Create report dir if report_dir is set
    if 'output_path' in args and args.output_path and not os.path.exists(
            args.output_path):
        os.makedirs(args.output_path)

    if export == 'html':
        output_path = os.path.abspath(args.output_path)

        LOG.info("Generating HTML output files to file://%s/index.html",
                 output_path)

        report_to_html.convert(
            all_reports,
            output_path,
            file_stats,
            file_checkers_map,
            file_runtime_map,
            file_failed_checkers_map)

        print("\n----==== Summary ====----")
        print("Number of analyzed analyzer result files: ",
              statistics.num_of_analyzer_result_files)
        print("Number of analyzer reports:               ",
              statistics.num_of_reports)
        print("Number of analyzer reports with source:   ",
              statistics.num_of_reports_with_source)

        print("\nHTML files were generated in '%s' folder." % output_path)
        return

    if export == 'gerrit':
        output_path = os.path.abspath(args.output_path) \
            if 'output_path' in args else None

        LOG.info("Generating Gerrit review files")

        gerrit.convert(all_reports, args.trim_path_prefix[0], output_path)

        print("\n----==== Summary ====----")
        print("Number of analyzed analyzer result files: ",
              statistics.num_of_analyzer_result_files)
        print("Number of analyzer reports:               ",
              statistics.num_of_reports)
        print("Number of analyzer reports with source:   ",
              statistics.num_of_reports_with_source)

        if output_path:
            print("\nGerrit review file was generated in '%s' folder." %
                  output_path)
        return

    if export == 'json':
        output_path = os.path.abspath(args.output_path)
        LOG.info("Generating JSON output files to '%s'", output_path)

        report_to_json.convert(
            all_reports,
            output_path,
            file_stats,
            file_checkers_map,
            file_runtime_map,
            file_failed_checkers_map)

        print("\n----==== Summary ====----")
        print("Number of analyzed analyzer result files: ",
              statistics.num_of_analyzer_result_files)
        print("Number of analyzer reports:               ",
              statistics.num_of_reports)
        print("Number of analyzer reports with source:   ",
              statistics.num_of_reports_with_source)

        print("\nJSON files were generated in '%s' folder." % output_path)
        return

    if export == 'codeclimate':
        output_path = os.path.abspath(args.output_path)
        LOG.info("Generating Code Climate output files to '%s'", output_path)
        codeclimate.convert(all_reports, output_path)

        print("\n----==== Summary ====----")
        print("Number of analyzed analyzer result files: ",
              statistics.num_of_analyzer_result_files)
        print("Number of analyzer reports:               ",
              statistics.num_of_reports)

        print("\nCode Climate files were generated in '%s' folder." %
              output_path)
        return

    if export == 'baseline':
        output_path = os.path.abspath(args.output_path) \
            if 'output_path' in args else None

        LOG.info("Generating baseline output")
        baseline.convert(all_reports, output_path)

        print("\n----==== Summary ====----")
        print("Number of analyzed analyzer result files: ",
              statistics.num_of_analyzer_result_files)
        print("Number of analyzer reports:               ",
              statistics.num_of_reports)

        if output_path:
            print("\nBaseline file was generated in '%s' folder." % output_path)
        return

    if 'create_suppress' in args:
        reports_helper.create_suppress_file(all_reports, args.suppress)
        sys.exit(0)

    reports_helper.dump_report_stats(statistics, file_stats, 'report')

    print("\n----==== Summary ====----")
    print("Number of processed analyzer result files: ",
          statistics.num_of_analyzer_result_files)
    print("Number of analyzer reports:                ",
          statistics.num_of_reports)
    print("Number of analyzer reports with source:    ",
          statistics.num_of_reports_with_source)

    if all_reports:
        print("\n----==== Checker Statistics ====----")
        statistics.print_checker_statistics()

    # Print reports.
    if all_reports:
        for report in all_reports:
            if 'print_steps' in args:
                report.print_steps()
            else:
                print(report)
    else:
        print("\nNo report data was found.")
        if 'review_status' in args:
            print("Review status(es) given were: " +
                  ', '.join(args.review_status))
derModel.from_pretrained(
            "naver-clova-ix/donut-base-finetuned-docvqa",
            revision="official"
        ).to(device)
        
        # Validate with test image
        test_img = Image.new('RGB', (300, 300), color=(255,255,255))
        pixel_values = processor(test_img, return_tensors="pt").pixel_values
        output = model.generate(pixel_values.to(device), max_length=5)
        decoded = processor.batch_decode(output)[0]
        
        if not decoded:
            raise ValueError("Donut validation failed")
            
        return processor, model
        
    except Exception as e:
        logging.critical(f"Donut initialization failed: {str(e)}")
        return None, None

def process_pdf_batch(batch):
    """Process batch of PDFs in parallel with document-level processing"""
    with ProcessPoolExecutor(max_workers=min(len(batch), 4)) as executor:
        futures = {executor.submit(process_pdf, pdf): pdf for pdf in batch}
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                yield result
            except Exception as e:
                logging.error(f"Processing failed: {str(e)}")

def process_page(page):
    """Process individual page with error handling"""
    try:
        text = page.extract_text()
        if len(text) > 50:
            return str(text)  # Ensure string conversion
        img = page.to_image(resolution=300).original
        ocr_result = ocr_utils.hybrid_ocr(img)
        return str(ocr_result) if ocr_result else ""
    except Exception as e:
        logging.error(f"Page processing error: {str(e)}")
        return ""

# Combined model loader with progress
@st.cache_resource(show_spinner=False)
def load_models():
    with st.spinner("Loading medical AI models..."):
        try:
            # Initialize BioBERT components separately
            from transformers import AutoTokenizer, AutoModelForTokenClassification
            biobert_tokenizer = AutoTokenizer.from_pretrained(
                "dmis-lab/biobert-base-cased-v1.1",
                truncation=True,
                max_length=512
            )
            biobert_model = AutoModelForTokenClassification.from_pretrained(
                "dmis-lab/biobert-base-cased-v1.1"
            )

            return {
                "biobert": pipeline(
                    "ner",
                    model=biobert_model,
                    tokenizer=biobert_tokenizer,
                    device=0 if torch.cuda.is_available() else -1,
                    aggregation_strategy="simple"
                ),
                "radbert": pipeline(
                    "text-classification",
                    model="zzxslp/RadBERT-RoBERTa-4m",
                    device=0 if torch.cuda.is_available() else -1
                ),
                "chatbot": pipeline(
                    "text-generation",
                    model="Mohammed-Altaf/Medical-ChatBot",
                    device=0 if torch.cuda.is_available() else -1,
                    max_length=200,
                    temperature=0.75
                ),
                "biobert_fr": pipeline(
                    "ner",
                    model="Dr-BERT/DrBERT-7GB",
                    device=0 if torch.cuda.is_available() else -1,
                    aggregation_strategy="simple"
                ),
                "medical_ner": pipeline(
                    "ner",
                    model="d4data/biomedical-ner-all",
                    device=0 if torch.cuda.is_available() else -1,
                    aggregation_strategy="simple",
                    framework="pt"
                )
            }
        except Exception as e:
            st.error(f"Model loading failed: {str(e)}")
            raise

# Load only needed models
models = load_models()
diagnosis_pipeline = models["radbert"]
chatbot_pipeline = models["chatbot"]

def plot_birads_distribution(df):
    """Interactive BI-RADS distribution visualization"""
    birads_counts = df['birads_score'].value_counts().reset_index()
    birads_counts.columns = ['BI-RADS Category', 'Count']
    
    fig = px.bar(
        birads_counts,
        x='BI-RADS Category',
        y='Count',
        color='BI-RADS Category',
        title="BI-RADS Category Distribution",
        labels={'Count': 'Number of Cases'},
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_findings_analysis(df):
    """Interactive findings analysis visualization"""
    findings_text = ' '.join(df['mammogram_results'].dropna())
    word_freq = pd.Series(findings_text.lower().split()).value_counts().reset_index()
    word_freq.columns = ['Term', 'Count']
    word_freq = word_freq[~word_freq['Term'].isin(STOP_WORDS)]
    
    fig = px.pie(
        word_freq.head(10),
        names='Term',
        values='Count',
        title="Top 10 Clinical Findings Terms",
        hole=0.3
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_temporal_trends(df):
    """Interactive temporal trends visualization"""
    df['exam_date'] = pd.to_datetime(df['exam_date'])
    df['year'] = df['exam_date'].dt.year
    
    # Year selection slider
    years = sorted(df['year'].unique())
    selected_year = st.slider(
        "Select Year Range",
        min_value=min(years),
        max_value=max(years),
        value=(min(years), max(years))
    )
    
    # Filter data
    filtered = df[(df['year'] >= selected_year[0]) & (df['year'] <= selected_year[1])]
    monthly_counts = filtered.resample('M', on='exam_date').size().reset_index(name='count')
    
    # Create interactive plot
    fig = px.line(
        monthly_counts,
        x='exam_date',
        y='count',
        title=f"Exam Trends {selected_year[0]}-{selected_year[1]}",
        labels={'exam_date': 'Date', 'count': 'Number of Exams'},
        markers=True
    )
    fig.update_layout(
        hovermode="x unified",
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True)
    )
    st.plotly_chart(fig, use_container_width=True)

# Create main tabs
tab1, tab2, tab3 = st.tabs(["OCR Processing", "Data Analysis", "Chatbot"])

with tab1:
    st.header("OCR Processing")
    st.write("Upload your scanned mammogram reports (PDFs) to extract data.")
    
    # PDF Uploader
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type="pdf",
        accept_multiple_files=True
    )
    
    # Add to OCR Processing tab
    psm_mode = st.selectbox(
        "Layout Analysis Mode",
        options=[
            ("Fully Automatic", "3"),
            ("Single Column", "4"), 
            ("Single Text Block", "6")
        ],
        format_func=lambda x: x[0],
        help="Adjust for different document layouts"
    )
    os.environ["TESS_PSM"] = psm_mode[1]
    
    if st.button("Process PDFs"):
        if not uploaded_files:
            st.warning("Please upload at least one PDF file")
        else:
            with st.spinner("Processing PDFs..."):
                init_db()
                
                # Create progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Process each PDF as a complete document
                all_results = []
                for i, file in enumerate(uploaded_files):
                    status_text.text(f"Processing {file.name}...")
                    
                    # Process the PDF
                    structured_data, metadata = process_pdf(file)
                    
                    # Add to results
                    all_results.append(structured_data)
                    
                    # Update progress
                    progress = (i + 1) / len(uploaded_files)
                    progress_bar.progress(progress)
                
                # Create DataFrame with one row per PDF
                if all_results:
                    # Ensure consistent columns
                    required_columns = [
                        'birads_score', 'document_date', 'document_type', 
                        'electronically_signed_by', 'exam_date', 'impression_result', 
                        'mammogram_results', 'patient_history', 'patient_name', 
                        'recommendation', 'testing_provider', 'ultrasound_results'
                    ]
                    
                    # Create DataFrame with consistent columns
                    df = pd.DataFrame(all_results)
                    
                    # Add missing columns with default values
                    for col in required_columns:
                        if col not in df.columns:
                            df[col] = "Unknown"
                    
                    # Store in session state
                    st.session_state.df = df
                    
                    # Display results
                    st.success(f"Processed {len(df)} documents")
                    st.dataframe(df)
                    
                    # Download button
                    st.download_button(
                        "Download Results",
                        df.to_csv(index=False),
                        "medical_data.csv",
                        "text/csv"
                    )
                else:
                    st.warning("No valid data extracted from PDFs")
    
    # Display results
    if 'df' in st.session_state:
        st.subheader("Processed Results")
        
        # Display structured data only if we have results
        st.subheader("Structured Medical Data")
        if not st.session_state['df'].empty:
            structured_data = ocr_utils.convert_to_structured_json(st.session_state['df'].iloc[0].to_dict())  # Show first result
            st.json(structured_data)
        else:
            st.warning("No structured data available - DataFrame is empty")
        
        # Display full dataframe with all fields
        st.subheader("Full Extracted Data")
        st.dataframe(st.session_state['df'], use_container_width=True)
        
        # Enhanced CSV download
        csv = st.session_state['df'].to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download Full Report",
            data=csv,
            file_name="full_mammogram_report.csv",
            mime="text/csv",
            help="Includes all extracted fields and confidence scores"
        )

with tab2:
    st.header("Data Analysis")
    st.write("Analyze the extracted data or upload your own CSV/Excel file.")
    
    # Data input handling
    if 'df' in st.session_state:
        df = st.session_state['df']
        st.success("Using data from OCR Processing tab.")
    else:
        uploaded_file = st.file_uploader(
            "Upload CSV or Excel file", 
            type=["csv", "xlsx"],
            help="Upload your own medical data for analysis"
        )
        df = pd.read_csv(uploaded_file) if uploaded_file and uploaded_file.name.endswith('.csv') else \
             pd.read_excel(uploaded_file) if uploaded_file and uploaded_file.name.endswith('.xlsx') else None

    if df is not None:
        # Data Validation Section
        st.subheader("Data Quality Check")
        
        # Validate dates
        invalid_dates = []
        if 'exam_date' in df.columns:
            for idx, date_str in df['exam_date'].items():
                try:
                    pd.to_datetime(date_str)
                except:
                    invalid_dates.append((idx, date_str))
                    df.at[idx, 'exam_date'] = "Not Available"
        
        # Validate BI-RADS scores
        invalid_birads = []
        birads_pattern = re.compile(r'BIRADS[\s-]*([0-6])', re.IGNORECASE)
        if 'birads_score' in df.columns:
            for idx, score in df['birads_score'].items():
                if not birads_pattern.match(str(score)):
                    invalid_birads.append((idx, score))
                    df.at[idx, 'birads_score'] = "Not Available"

        # Show validation results
        col1, col2 = st.columns(2)
        with col1:
            if invalid_dates:
                st.error(f"⚠️ Found {len(invalid_dates)} invalid dates")
                if st.checkbox("Show invalid date details"):
                    st.write(pd.DataFrame(invalid_dates, columns=["Row", "Invalid Date"]))
        
        with col2:
            if invalid_birads:
                st.error(f"⚠️ Found {len(invalid_birads)} invalid BI-RADS scores")
                if st.checkbox("Show invalid BI-RADS details"):
                    st.write(pd.DataFrame(invalid_birads, columns=["Row", "Invalid Score"]))

        # Enhanced Visualization
        st.subheader("Clinical Findings Analysis")
        tab1, tab2, tab3 = st.tabs(["BI-RADS Distribution", "Findings Analysis", "Temporal Trends"])
        
        with tab1:
            if 'birads_score' in df.columns:
                plot_birads_distribution(df)
            else:
                st.warning("BI-RADS data not available for visualization")

        with tab2:
            if 'mammogram_results' in df.columns:
                plot_findings_analysis(df)
            else:
                st.warning("Findings data not available for visualization")

        with tab3:
            if 'exam_date' in df.columns:
                plot_temporal_trends(df)
            else:
                st.warning("Date data not available for temporal analysis")

        # Basic statistics
        st.subheader("Basic Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            if 'patient_name' in df.columns:
                unique_patients = df['patient_name'].nunique()
                st.metric("Unique Patients", unique_patients)
            else:
                st.metric("Unique Patients", "N/A")
        with col2:
            total_reports = len(df)
            st.metric("Total Reports", total_reports)
        with col3:
            latest_date = df['exam_date'].max() if 'exam_date' in df.columns else "N/A"
            st.metric("Latest Exam Date", latest_date)

        # Exam type analysis
        if 'document_type' in df.columns:
            st.subheader("Exam Type Distribution")
            exam_dist = df['document_type'].value_counts().reset_index()
            exam_dist.columns = ['Exam Type', 'Count']
            st.bar_chart(exam_dist.set_index('Exam Type'))

        # BIRADS analysis
        if 'birads_score' in df.columns:
            st.subheader("BIRADS Analysis")
            
            # Interactive BIRADS score filter
            selected_scores = st.multiselect(
                "Select BIRADS scores to filter:",
                options=['0', '1', '2', '3', '4', '5'],
                default=['4', '5'],
                help="Filter cases by BIRADS score"
            )
            
            # Filter dataframe
            filtered_df = df[df['birads_score'].isin(selected_scores)]
            
            if not filtered_df.empty:
                # Display filtered cases
                st.write(f"Showing {len(filtered_df)} cases with BIRADS {', '.join(selected_scores)}")
                st.dataframe(filtered_df[['patient_name', 'birads_score', 'exam_date']], use_container_width=True)
                
                # Correlation heatmap
                st.subheader("Feature Correlations")
                try:
                    numeric_cols = df.select_dtypes(include=['number']).columns
                    if len(numeric_cols) > 1:
                        corr_matrix = df[numeric_cols].corr()
                        fig = px.imshow(
                            corr_matrix,
                            labels=dict(x="Features", y="Features", color="Correlation"),
                            x=numeric_cols,
                            y=numeric_cols,
                            color_continuous_scale='RdBu',
                            zmin=-1,
                            zmax=1
                        )
                        fig.update_layout(title="Correlation Heatmap of Numeric Features")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Not enough numeric columns for correlation analysis")
                except Exception as e:
                    st.warning(f"Could not generate heatmap: {str(e)}")
            else:
                st.info("No cases found with selected BIRADS scores")

        # Findings analysis
        if 'mammogram_results' in df.columns:
            st.subheader("Common Findings")
            
            def convert_to_str(item):
                """Safely convert findings data to strings"""
                if isinstance(item, dict):
                    return item.get('description', '')  # Match our data structure
                elif isinstance(item, list):
                    return ' '.join([convert_to_str(i) for i in item])
                elif isinstance(item, str):
                    return item
                else:  # Handle numbers/other types
                    return str(item)

            # Process findings with type safety
            with st.spinner("Processing clinical findings..."):
                try:
                    # Convert all findings to cleaned strings
                    df['mammogram_results'] = df['mammogram_results'].apply(
                        lambda x: convert_to_str(ocr_utils.extract_findings_text(x))
                    )
                    
                    # Debug type distribution if needed
                    if st.checkbox("Show findings type debug info"):
                        st.write("Findings type distribution:")
                        st.write(df['mammogram_results'].apply(type).value_counts())
                        st.write("Sample processed findings:")
                        st.write(df['mammogram_results'].head(3).to_dict())
                    
                    # Now safely join validated strings
                    findings_text = ' '.join(
                        df['mammogram_results'].dropna().astype(str)
                    )
                except Exception as e:
                    st.error(f"Failed to process findings: {str(e)}")
                    logging.error(f"Findings processing failed: {str(e)}")
                    findings_text = ""

            if findings_text.strip():
                # Clean and analyze text
                findings_text = re.sub(r'[^\w\s.-]', '', findings_text)  # Remove special chars
                word_freq = pd.Series(findings_text.lower().split()).value_counts()[:10]
                
                # Display results
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.bar_chart(word_freq)
                with col2:
                    st.metric("Unique Terms Found", len(word_freq))
            else:
                st.write("No analyzable findings data available")

        # Interactive exam trends over time
        if 'exam_date' in df.columns:
            st.subheader("Exam Trends Over Time")
            try:
                df['exam_date'] = pd.to_datetime(df['exam_date'])
                df['year'] = df['exam_date'].dt.year
                
                # Year selection slider
                years = sorted(df['year'].unique())
                selected_year = st.slider(
                    "Select Year Range",
                    min_value=min(years),
                    max_value=max(years),
                    value=(min(years), max(years))
                )
                
                # Filter data
                filtered = df[(df['year'] >= selected_year[0]) & (df['year'] <= selected_year[1])]
                monthly_counts = filtered.resample('M', on='exam_date').size().reset_index(name='count')
                
                # Create interactive plot
                fig = px.line(
                    monthly_counts,
                    x='exam_date',
                    y='count',
                    title=f"Exam Trends {selected_year[0]}-{selected_year[1]}",
                    labels={'exam_date': 'Date', 'count': 'Number of Exams'},
                    markers=True
                )
                fig.update_layout(
                    hovermode="x unified",
                    xaxis=dict(showgrid=True),
                    yaxis=dict(showgrid=True)
                )
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.warning(f"Could not analyze trends: {str(e)}")

        # Interactive BIRADS Score Distribution
        st.subheader("BIRADS Score Distribution")
        if 'birads_score' in df.columns:
            try:
                birads_counts = df['birads_score'].value_counts().reset_index()
                birads_counts.columns = ['BIRADS Score', 'Count']
                
                fig = px.bar(
                    birads_counts,
                    x='BIRADS Score',
                    y='Count',
                    color='BIRADS Score',
                    title='BIRADS Score Distribution',
                    labels={'Count': 'Number of Cases'},
                    color_discrete_sequence=px.colors.sequential.Plasma
                )
                fig.update_layout(
                    xaxis={'categoryorder':'total descending'},
                    hovermode='closest'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Add summary statistics
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Cases", len(df))
                col2.metric("Most Common Score", birads_counts.iloc[0]['BIRADS Score'])
                col3.metric("Highest Risk Cases", len(df[df['birads_score'].isin(['4', '5'])]))
                
            except Exception as e:
                st.warning(f"Could not analyze BIRADS scores: {str(e)}")
        else:
            st.warning("BIRADS score data not available for visualization")

        # Additional Information Analysis
        st.subheader("Findings Analysis")
        if 'impression_result' in df.columns:
            try:
                # Generate word cloud
                text = ' '.join(df['impression_result'].dropna())
                wordcloud = WordCloud(width=800, height=400).generate(text)
                
                # Display using matplotlib
                fig, ax = plt.subplots()
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis("off")
                st.pyplot(fig)
                
                # Show frequent terms
                st.write("**Common Clinical Terms**")
                from collections import Counter
                words = re.findall(r'\b[A-Za-z]{4,}\b', text)
                counter = Counter([w.lower() for w in words if w.lower() not in STOP_WORDS])
                st.write(pd.DataFrame(counter.most_common(10), columns=['Term', 'Count']))
                
            except Exception as e:
                st.error(f"Text analysis failed: {str(e)}")

    else:
        st.info("ℹ️ No data available. Process PDFs or upload a file to begin analysis.")

with tab3:
    st.header("Chatbot")
    st.write("Ask questions or upload a CSV for analysis.")
    
    # CSV analysis upload
    csv_file = st.file_uploader(
        "Upload CSV for analysis",
        type="csv",
        key="chat_csv",
        help="Upload patient data for contextual analysis"
    )
    
    # Initialize chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg['role']):
            st.markdown(msg['content'])
    
    # Handle CSV analysis
    if csv_file:
        with st.spinner("Analyzing patient data..."):
            try:
                df = pd.read_csv(csv_file)
                analysis_results = chatbot_utils.analyze_csv(df, diagnosis_pipeline)
                
                # Format analysis results
                report = "📊 **Patient Data Analysis Report**\n\n"
                report += f"• Total patients: {len(df)}\n"
                report += f"• Key findings: {analysis_results[:3]}\n"  # Show top 3 findings
                report += "\nAsk me anything about this data!"
                
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'content': report
                })
                st.rerun()
                
            except Exception as e:
                st.error(f"CSV processing failed: {str(e)}")
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'content': "❌ Failed to analyze CSV. Please check the file format."
                })
    
    # Handle user input
    user_input = st.chat_input("Type your message here...")
    if user_input:
        context = ""
        if csv_file:
            context = f"Analyzing {len(df)} patient records. "
        elif 'df' in st.session_state:
            context += f"Report contains: {st.session_state['df']['impression_result'].iloc[0][:200]}... "
        
        # Add BIRADS context
        if 'birads_score' in st.session_state.get('df', pd.DataFrame()).columns:
            context += f"Current BIRADS score: {st.session_state['df']['birads_score'].mode()[0]}. "
        
        full_query = f"{context}\n\nUser question: {user_input}"
        
        # Add user message to history
        st.session_state.chat_history.append({'role': 'user', 'content': user_input})
        
        # Generate response with medical context
        response = chatbot_pipeline(
            full_query,
            max_length=300,
            temperature=0.7,
            num_return_sequences=1
        )[0]['generated_text']
        
        st.session_state.chat_history.append({'role': 'assistant', 'content': response})
        st.rerun()

# Add footer with resource links
# Cache status display
st.sidebar.markdown("### Cache Status")
st.sidebar.write(f"Preprocessed Images: {len(ocr_utils.IMAGE_CACHE)}")
st.sidebar.write(f"OCR Results: {len(ocr_utils.OCR_CACHE) if hasattr(ocr_utils, 'OCR_CACHE') else 0}")
st.sidebar.write(f"LLM Responses: {len(ocr_utils.LLM_CACHE) if hasattr(ocr_utils, 'LLM_CACHE') else 0}")

st.sidebar.markdown("### Resources")
st.sidebar.markdown("- [Clinical Guidelines](https://example.com)")
st.sidebar.markdown("- [Medical Knowledge Base](https://example.com)")
st.sidebar.markdown("- [Emergency Protocols](https://example.com)")

# Database functions
def init_db():
    """Initialize SQLite database with schema"""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS reports (
                md5_hash TEXT PRIMARY KEY,
                filename TEXT,
                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                patient_name TEXT,
                exam_date TEXT,
                findings TEXT,
                metadata TEXT
            )
        """)

def save_to_db(data: dict):
    """Save processed results to database"""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT OR REPLACE INTO reports 
            (md5_hash, filename, patient_name, exam_date, findings, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            data['md5_hash'],
            data['filename'],
            data.get('patient_name'),
            data.get('exam_date'),
            json.dumps(data.get('findings')),
            json.dumps(data.get('metadata'))
        ))

# Add to sidebar section at bottom of file
st.sidebar.title("Mammogram Analysis Dashboard")
st.sidebar.markdown("""
**Clinical Decision Support System**  
AI-powered analysis of mammogram reports with:
- PDF OCR extraction
- Medical NLP processing
- BIRADS classification
- Clinical insights generation
""")

@st.cache_resource
def load_translation_model():
    return pipeline(
        "translation_fr_to_en",
        model="Helsinki-NLP/opus-mt-fr-en",
        device=0 if torch.cuda.is_available() else -1
    )

if __name__ == "__main__":
    st.write("Medical AI Assistant is running...") 
