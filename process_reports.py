#!/usr/bin/env python3
"""
Medical Report Processor CLI

This script provides a command-line interface for processing medical reports,
extracting structured information, and validating the results.
"""

import os
import sys
import argparse
import logging
import pandas as pd
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("processing.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Add the current directory to path to ensure imports work
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

try:
    from src.document_processing.main_processor import (
        process_single_report,
        process_directory,
        fix_dataframe
    )
except ImportError as e:
    logger.error(f"Error importing required modules: {e}")
    logger.error("Make sure you're running this script from the project root directory")
    sys.exit(1)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Process medical reports and extract structured information'
    )
    
    # Input arguments
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--file', '-f',
        help='Path to a single PDF report to process'
    )
    input_group.add_argument(
        '--directory', '-d',
        help='Path to a directory containing PDF reports to process'
    )
    input_group.add_argument(
        '--fix-dataframe', '-x',
        help='Path to a CSV file containing previously processed results to fix'
    )
    
    # Output arguments
    parser.add_argument(
        '--output', '-o',
        help='Path to save output CSV file (default: output.csv)',
        default='output.csv'
    )
    
    # Processing options
    parser.add_argument(
        '--no-llm',
        action='store_true',
        help='Disable LLM-based enhancement'
    )
    parser.add_argument(
        '--raw-text-dir',
        help='Directory containing raw OCR text files (for fixing dataframes)'
    )
    
    return parser.parse_args()

def process_file(file_path, use_llm=True):
    """Process a single PDF file."""
    logger.info(f"Processing file: {file_path}")
    
    try:
        result = process_single_report(file_path, use_llm)
        if result.get('processing_status') == 'error':
            logger.error(f"Error processing {file_path}: {result.get('error_message')}")
        else:
            logger.info(f"Successfully processed {file_path}")
        
        return pd.DataFrame([result])
    except Exception as e:
        logger.error(f"Failed to process {file_path}: {e}")
        return pd.DataFrame([{
            'source_file': os.path.basename(file_path),
            'processing_status': 'error',
            'error_message': str(e)
        }])

def process_dir(dir_path, output_path, use_llm=True):
    """Process all PDF files in a directory."""
    logger.info(f"Processing directory: {dir_path}")
    
    try:
        df = process_directory(dir_path, output_path, use_llm)
        logger.info(f"Processed {len(df)} files")
        return df
    except Exception as e:
        logger.error(f"Failed to process directory {dir_path}: {e}")
        return pd.DataFrame()

def fix_existing_dataframe(csv_path, output_path, raw_text_dir=None):
    """Fix and enhance an existing dataframe."""
    logger.info(f"Fixing dataframe: {csv_path}")
    
    try:
        # Load the CSV file
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded dataframe with {len(df)} rows and {len(df.columns)} columns")
        
        # Load raw texts if directory is provided
        raw_texts = None
        if raw_text_dir:
            raw_texts = {}
            logger.info(f"Loading raw texts from {raw_text_dir}")
            try:
                for file in os.listdir(raw_text_dir):
                    if file.endswith('.txt'):
                        file_path = os.path.join(raw_text_dir, file)
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                raw_texts[file.replace('.txt', '.pdf')] = f.read()
                        except Exception as e:
                            logger.error(f"Error reading {file_path}: {e}")
                logger.info(f"Loaded {len(raw_texts)} raw text files")
            except Exception as e:
                logger.error(f"Error loading raw texts: {e}")
        
        # Fix the dataframe
        fixed_df = fix_dataframe(df, raw_texts)
        
        # Save the fixed dataframe
        fixed_df.to_csv(output_path, index=False)
        logger.info(f"Fixed dataframe saved to {output_path}")
        
        return fixed_df
    except Exception as e:
        logger.error(f"Failed to fix dataframe {csv_path}: {e}")
        return pd.DataFrame()

def main():
    """Main function."""
    args = parse_args()
    
    try:
        if args.file:
            df = process_file(args.file, not args.no_llm)
            df.to_csv(args.output, index=False)
            print(f"Results saved to {args.output}")
        
        elif args.directory:
            df = process_dir(args.directory, args.output, not args.no_llm)
            print(f"Processed {len(df)} files. Results saved to {args.output}")
        
        elif args.fix_dataframe:
            df = fix_existing_dataframe(args.fix_dataframe, args.output, args.raw_text_dir)
            print(f"Fixed dataframe with {len(df)} rows. Results saved to {args.output}")
        
        # Print a summary of the processing
        if not df.empty:
            success_count = len(df[df['processing_status'] == 'success']) if 'processing_status' in df.columns else 0
            error_count = len(df[df['processing_status'] == 'error']) if 'processing_status' in df.columns else 0
            
            print(f"\nProcessing Summary:")
            print(f"  Total files: {len(df)}")
            print(f"  Successfully processed: {success_count}")
            print(f"  Errors: {error_count}")
            
            # Check for missing critical fields
            critical_fields = ["birads_score", "impression", "exam_date", "exam_type"]
            for field in critical_fields:
                if field in df.columns:
                    missing = df[df[field].isin(["", "Not Available"])].shape[0]
                    print(f"  Missing {field}: {missing} ({(missing/len(df))*100:.1f}%)")
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"An error occurred: {e}")
        print("Check the processing.log file for details")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 