import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*missing ScriptRunContext.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*device_map.*")
warnings.filterwarnings("ignore", category=UserWarning)  # Suppress all UserWarnings

# Custom exception class
class OCRError(Exception):
    """Custom exception for OCR processing errors"""
    pass

# Import Streamlit first
import streamlit as st
import time
import platform
import os
import psutil
import json
import pdfplumber
import hashlib
from concurrent.futures import ProcessPoolExecutor
import sqlite3
from pathlib import Path

# Database configuration
DB_PATH = "mammo_reports.db"
BATCH_SIZE = 10

def manage_caches():
    """Auto-clear caches when memory usage exceeds 80%"""
    process = psutil.Process()
    mem_info = process.memory_info()
    
    if mem_info.rss / (1024 ** 3) > 0.8:  # 0.8GB threshold
        ocr_utils.IMAGE_CACHE.clear()
        ocr_utils.OCR_CACHE.clear()
        st.rerun()
# import streamlit_authenticator as stauth  # <- Comment this

# Set page config immediately after import
st.set_page_config(
    page_title="Mammo AI Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Then other imports
import pandas as pd
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers.pipelines import pipeline
from pdf2image import convert_from_path, convert_from_bytes
from PIL import Image
import cv2
import pytesseract
import ocr_utils, chatbot_utils
from huggingface_hub import login
import plotly.express as px
from transformers import MarianTokenizer, MarianMTModel, AutoModel, AutoProcessor, VisionEncoderDecoderModel, DonutProcessor
import concurrent.futures
import logging
from functools import partial
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from nltk.corpus import stopwords
from groq import Groq

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
    from paddleocr import PaddleOCR
    return PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)

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
            
        model = VisionEncoderDecoderModel.from_pretrained(
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
    """Process 10 PDFs in parallel with mixed text/OCR"""
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

def process_pdf(uploaded_file):
    """Process PDF with parallel page processing"""
    manage_caches()  # Check memory before processing
    file_hash = hashlib.md5(uploaded_file.getvalue()).hexdigest()
    
    # Check cache
    if Path(DB_PATH).exists():
        with sqlite3.connect(DB_PATH) as conn:
            cached = conn.execute("""
                SELECT findings, metadata FROM reports
                WHERE md5_hash = ?
            """, (file_hash,)).fetchone()
            if cached:
                return json.loads(cached[0]), json.loads(cached[1])

    # Parallel page processing
    text_content = []
    with pdfplumber.open(uploaded_file) as pdf:
        with ThreadPoolExecutor(max_workers=min(4, os.cpu_count())) as executor:
            page_futures = {
                executor.submit(process_page, page): page
                for page in pdf.pages
            }
            for future in concurrent.futures.as_completed(page_futures):
                try:
                    page_text = future.result()
                    text_content.append(page_text)
                except Exception as e:
                    logging.error(f"Page failed: {str(e)}")

    # Process and cache
    structured_data = ocr_utils.process_text("\n".join(text_content))
    metadata = {
        "md5_hash": file_hash,
        "filename": uploaded_file.name,
        "pages": len(text_content)
    }
    
    save_to_db({
        **structured_data,
        "metadata": metadata
    })
    
    return structured_data, metadata

def process_single_page(image, page_num, uploaded_file):
    try:
        # Simple OCR processing
        ocr = init_paddle()
        result = ocr.ocr(np.array(image), cls=True)
        text = ' '.join([line[1][0] for line in result[0]]) if result else ""
        
        # Basic field extraction
        return {
            'text': text,
            'source_pdf': uploaded_file.name,
            'page_number': page_num + 1
        }
    except Exception as e:
        return {'error': str(e)}

        # Add error handling for empty OCR results
        if not text:
            logging.warning(f"No text extracted from page {page_num}")
            structured_data = ocr_utils.default_structured_output()
            warnings = ['OCR failed - no text detected']
            return {
                **structured_data,
                'source_pdf': uploaded_file.name,
                'page_number': page_num + 1,
                'warnings': warnings
            }

        # Enhanced Mixtral prompt with layout context
        prompt = f"""MEDICAL DOCUMENT STRUCTURE:
        No layout data available

        OCR TEXT:
        {text[:3000]}

        TASK: Extract and validate these fields:
        1. patient_name (title case)
        2. exam_date (YYYY-MM-DD)
        3. birads_right (0-6)
        4. birads_left (0-6)
        5. impressions (markdown bullets)
        6. findings (structured list)
        7. recommendations (numbered list)
        8. clinical_history (paragraph)
        9. document_date (from header/footer)
        10. electronically_signed_by

        RULES:
        - Correct OCR errors using medical context
        - Handle both English/French terms
        - Return JSON with confidence scores
        - Mark missing fields as 'Unknown'
        
        EXAMPLE OUTPUT:
        {{
            "patient_name": "Jane Doe",
            "exam_date": "2021-12-23",
            "birads_right": 1,
            "clinical_history": "Breast augmentation in 2010...",
            "impressions": ["No malignancy", "ACR Category A"],
            "confidence_scores": {{
                "patient_name": 0.95,
                "exam_date": 0.90
            }}
        }}"""
        
        # Context correction with Mixtral 8x7B
        groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        try:
            response = groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="mixtral-8x7b-32768",
                temperature=0.1,
                max_tokens=2000,
                response_format={"type": "json_object"}
            )
            
            structured_data = json.loads(response.choices[0].message.content)
            warnings = [] if 'confidence' in structured_data else ['Low confidence flags']
            
        except Exception as e:
            logging.error(f"Mixtral processing failed: {str(e)}")
            structured_data = ocr_utils.default_structured_output()
            warnings = ['AI processing failed - using raw OCR']

        # Standardized data structure with safe field access
        data = {
            'patient_name': structured_data.get('patient_name', 'Unknown'),
            'document_date': structured_data.get('document_date', 'Unknown'),
            'exam_type': structured_data.get('exam_type', 'Unknown'),
            'exam_date': structured_data.get('exam_date', 'Unknown'),
            'clinical_history': structured_data.get('clinical_history', ''),
            'birads_right': structured_data.get('birads_right', 'Unknown'),
            'birads_left': structured_data.get('birads_left', 'Unknown'),
            'impressions': structured_data.get('impressions', 'Unknown'),
            'findings': ocr_utils.extract_findings_text(structured_data.get('findings', 'Unknown')),
            'follow-up_recommendation': structured_data.get('follow-up_recommendation', 'Unknown'),
            'source_pdf': uploaded_file.name,
            'page_number': page_num + 1,
            'warnings': ', '.join(warnings) if warnings else 'None',
            'processing_confidence': 0.0,
            'raw_ocr_text': text  # Direct text access
        }
        
        return {
            'patient_name': structured_data.get('patient_name', 'Unknown'),
            'exam_date': structured_data.get('exam_date', 'Unknown'),
            'birads_right': structured_data.get('birads_right', 'Unknown'),
            'birads_left': structured_data.get('birads_left', 'Unknown'),
            'impressions': structured_data.get('impressions', 'Not Available'),
            'findings': ocr_utils.extract_findings_text(structured_data.get('findings', '')),
            'source_pdf': uploaded_file.name,
            'page_number': page_num + 1,
            'processing_confidence': structured_data.get('confidence', {}).get('overall', 0.0)
        }
    
    except Exception as e:
        logging.error(f"Error processing page {page_num+1} of {uploaded_file.name}", exc_info=True)
        return {
            'error': f"Page {page_num+1}: {str(e)}",
            'source pdf': uploaded_file.name,
            'page number': page_num + 1
        }

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
    findings_text = ' '.join(df['findings'].dropna())
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
            with st.spinner("Processing PDFs in batches..."):
                init_db()
                total_files = len(uploaded_files)
                extracted_data = []
                error_messages = []
                template_warnings = []
                template_status = st.empty()  # Create placeholder for template status
                
                # Process in batches
                for i in range(0, total_files, BATCH_SIZE):
                    batch = uploaded_files[i:i+BATCH_SIZE]
                    batch_results = []
                    
                    for result in process_pdf_batch(batch):
                        batch_results.append(result)
                        
                    # Update progress
                    progress = min((i + len(batch)) / total_files, 1.0)
                    st.progress(progress)
                    
                    # Update session state
                    if batch_results:
                        new_df = pd.DataFrame(batch_results)
                        st.session_state['df'] = pd.concat([
                            st.session_state.get('df', pd.DataFrame()),
                            new_df
                        ])
                
                # Create progress bar
                progress_bar = st.progress(0)
                total_files = len(uploaded_files)
                
                # Initialize dataframe in session state if not exists
                if 'df' not in st.session_state:
                    st.session_state['df'] = pd.DataFrame()

                # Simple processing loop
                if uploaded_files:
                    all_text = []
                    for uploaded_file in uploaded_files:
                        with pdfplumber.open(uploaded_file) as pdf:
                            for page in pdf.pages:
                                all_text.append(page.extract_text())
                    
                    st.session_state['df'] = pd.DataFrame({
                        'text': all_text,
                        'source': [f.name for f in uploaded_files]
                    })

                    # Simple processing loop
                    if uploaded_files:
                        all_text = []
                        for uploaded_file in uploaded_files:
                            with pdfplumber.open(uploaded_file) as pdf:
                                for page in pdf.pages:
                                    all_text.append(page.extract_text())
                        processed_pages += 1
                        progress = processed_pages / total_pages
                        progress_bar.progress(min(progress, 1.0))
                        
                        try:
                            result = future.result()
                            if 'error' in result:
                                error_messages.append(result['error'])
                            else:
                                # Collect extracted data
                                extracted_data.append(result)
                                template_warnings.extend(result.get('template_warnings', []))
                        except Exception as e:
                            error_msg = f"Processing failed: {str(e)}"
                            logging.error(error_msg, exc_info=True)
                            error_messages.append(error_msg)
                
                template_status.empty()
                
                # Display template warnings
                if template_warnings:
                    unique_warnings = list(set(template_warnings))
                    with st.expander("⚠️ Template Matching Warnings"):
                        st.write("The following issues were detected during template matching:")
                        for warning in unique_warnings:
                            st.write(f"- {warning}")
                        st.info("Please verify extracted fields and consider updating template images")
                
                # Handle processing results
                if error_messages:
                    st.error("Some pages failed to process:")
                    for msg in error_messages[-3:]:  # Show last 3 errors
                        st.write(f"- {msg}")
                    st.info("Check mammo_ai.log for full details")
                
                # Create final dataframe after processing all pages
                if extracted_data:
                    try:
                        # Convert all entries to valid dicts with null checks
                        valid_data = []
                        for entry in extracted_data:
                            if 'error' not in entry and entry.get('birads_right') not in [None, 'Unknown']:
                                valid_data.append({
                                    'patient_name': entry.get('patient_name', 'Unknown'),
                                    'exam_date': entry.get('exam_date', 'Unknown'),
                                    'birads_right': int(entry.get('birads_right', 0)),
                                    'birads_left': int(entry.get('birads_left', 0)),
                                    'findings': entry.get('findings', ''),
                                    'source_pdf': entry.get('source_pdf', 'Unknown'),
                                    'page_number': int(entry.get('page_number', 0)),
                                    'processing_confidence': float(entry.get('processing_confidence', 0.0))
                                })
                        
                        if valid_data:
                            df = pd.DataFrame(valid_data).astype({
                                'birads_right': 'int8',
                                'birads_left': 'int8',
                                'processing_confidence': 'float32',
                                'page_number': 'int32'
                            })
                            df['exam_date'] = pd.to_datetime(df['exam_date'], errors='coerce').dt.date
                            st.session_state['df'] = df.dropna(subset=['patient_name', 'exam_date'])
                            st.success(f"Processed {len(valid_data)} valid pages")
                            st.dataframe(st.session_state['df'])
                            
                            # Show sample structured data
                            structured_sample = ocr_utils.convert_to_structured_json(
                                st.session_state['df'].iloc[0].to_dict()
                            )
                            st.json(structured_sample)
                        else:
                            st.warning("No valid data extracted from any pages")
                            st.session_state['df'] = pd.DataFrame()  # Ensure empty DF

                    except Exception as e:
                        st.error(f"Data formatting failed: {str(e)}")
                        logging.error(f"Data formatting error: {str(e)}", exc_info=True)
    
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
            if 'findings' in df.columns:
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
            if 'patient name' in df.columns:
                unique_patients = df['patient name'].nunique()
                st.metric("Unique Patients", unique_patients)
            else:
                st.metric("Unique Patients", "N/A")
        with col2:
            total_reports = len(df)
            st.metric("Total Reports", total_reports)
        with col3:
            latest_date = df['exam date'].max() if 'exam date' in df.columns else "N/A"
            st.metric("Latest Exam Date", latest_date)

        # Exam type analysis
        if 'exam type' in df.columns:
            st.subheader("Exam Type Distribution")
            exam_dist = df['exam type'].value_counts().reset_index()
            exam_dist.columns = ['Exam Type', 'Count']
            st.bar_chart(exam_dist.set_index('Exam Type'))

        # BIRADS analysis
        if {'birads_score', 'exam_date'}.issubset(df.columns):
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
        if 'findings' in df.columns:
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
                    df['findings'] = df['findings'].apply(
                        lambda x: convert_to_str(ocr_utils.extract_findings_text(x))
                    )
                    
                    # Debug type distribution if needed
                    if st.checkbox("Show findings type debug info"):
                        st.write("Findings type distribution:")
                        st.write(df['findings'].apply(type).value_counts())
                        st.write("Sample processed findings:")
                        st.write(df['findings'].head(3).to_dict())
                    
                    # Now safely join validated strings
                    findings_text = ' '.join(
                        df['findings'].dropna().astype(str)
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
        if 'additional_information' in df.columns:
            try:
                # Generate word cloud
                text = ' '.join(df['additional_information'].dropna())
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
        st.warning("Patient name column missing in data")
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
            context += f"Report contains: {st.session_state['df']['additional_information'].iloc[0][:200]}... "
        
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
st.sidebar.write(f"OCR Results: {len(ocr_utils.OCR_CACHE)}")
st.sidebar.write(f"LLM Responses: {len(ocr_utils.LLM_CACHE)}")

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
            json.dumps(data['metadata'])
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
