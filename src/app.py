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
from sqlalchemy import create_engine, Column, String, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from transformers import pipeline, DonutProcessor, VisionEncoderDecoderModel
import torch
from huggingface_hub import login
from groq import Groq
import ocr_utils
import chatbot_utils
from nltk.corpus import stopwords
from pathlib import Path

# Constants
DB_URL = "sqlite:///mammo_reports.db"  # SQLAlchemy connection string
engine = create_engine(DB_URL)
Base = declarative_base()
Session = sessionmaker(bind=engine)

# Define the database model
class Report(Base):
    __tablename__ = 'reports'
    
    md5_hash = Column(String, primary_key=True)
    filename = Column(String)
    processed_at = Column(DateTime, default=datetime.datetime.now)
    patient_name = Column(String)
    exam_date = Column(String)
    findings = Column(Text)
    report_metadata = Column(Text)
    raw_ocr_text = Column(Text)
    
    def __repr__(self):
        return f"<Report(filename='{self.filename}')>"

# Add this near the top of your app.py, before any other Streamlit calls
st.set_page_config(
    page_title="Medical Report Processor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.example.com/help',
        'Report a bug': "https://www.example.com/bug",
        'About': "# Medical Report Processor\nAI-powered analysis of medical reports."
    }
)

def init_db():
    """Initialize database with SQLAlchemy"""
    Base.metadata.create_all(engine)

def save_to_db(data: dict):
    """Save processed results to database using SQLAlchemy"""
    try:
        session = Session()
        
        # Create a new report object
        report = Report(
            md5_hash=data['md5_hash'],
            filename=data['filename'],
            patient_name=data.get('patient_name', 'Not Available'),
            exam_date=data.get('exam_date', 'Not Available'),
            findings=json.dumps(data.get('findings', {})),
            report_metadata=json.dumps(data.get('metadata', {})),
            raw_ocr_text=data.get('raw_ocr_text', '')
        )
        
        # Add and commit
        session.merge(report)  # Use merge instead of add to handle updates
        session.commit()
        logging.info(f"Saved report to database: {data['filename']}")
        
    except Exception as e:
        logging.error(f"Database error: {str(e)}")
        if session:
            session.rollback()
    finally:
        if session:
            session.close()

# Add a function to clear all caches
def clear_all_caches():
    """Clear all application caches"""
    if hasattr(ocr_utils, 'IMAGE_CACHE'):
        ocr_utils.IMAGE_CACHE.clear()
    if hasattr(ocr_utils, 'OCR_CACHE'):
        ocr_utils.OCR_CACHE.clear()
    if hasattr(ocr_utils, 'LLM_CACHE'):
        ocr_utils.LLM_CACHE.clear()
    st.cache_data.clear()
    st.cache_resource.clear()
    return True

def process_pdf(uploaded_file):
    """Process PDF as a complete document with combined text from all pages"""
    try:
        # Generate file hash for caching
        file_hash = hashlib.sha256(uploaded_file.getvalue()).hexdigest()
        
        # Check cache using SQLAlchemy
        try:
            session = Session()
            cached_report = session.query(Report).filter_by(md5_hash=file_hash).first()
            
            if cached_report:
                logging.info(f"Using cached results for {uploaded_file.name}")
                return json.loads(cached_report.findings), json.loads(cached_report.report_metadata)
                
        except Exception as e:
            logging.error(f"Cache lookup error: {str(e)}")
        finally:
            if 'session' in locals():
                session.close()
        
        # Extract text from all pages
        all_text = []
        with pdfplumber.open(uploaded_file) as pdf:
            # Process all pages and combine text
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    all_text.append(page_text)
                    logging.info(f"Extracted text from page {len(all_text)} using pdfplumber")
                else:
                    # Fallback to OCR if text extraction fails
                    img = page.to_image(resolution=300).original
                    ocr_text = ocr_utils.simple_ocr(np.array(img))
                    all_text.append(ocr_text)
                    logging.info(f"Extracted text from page {len(all_text)} using OCR fallback")
        
        # Combine all text into a single document
        combined_text = "\n\n".join(all_text)
        logging.info(f"Combined text length: {len(combined_text)} characters")
        logging.info(f"Text sample: {combined_text[:200]}...")
        
        # Process the combined text to extract all fields
        structured_data = ocr_utils.process_document_text(combined_text)
        
        # Debug output
        logging.info(f"Extraction results: {json.dumps({k: v for k, v in structured_data.items() if k != 'raw_ocr_text'})}")
        
        # Ensure raw OCR text is preserved
        structured_data['raw_ocr_text'] = combined_text
        
        # Add metadata
        metadata = {
            "md5_hash": file_hash,
            "filename": uploaded_file.name,
            "pages": len(all_text),
            "processing_date": pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            "is_deidentified": ocr_utils.is_deidentified_document(combined_text)
        }
        
        # Only try to save to DB if the function exists
        try:
            save_to_db({
                "md5_hash": file_hash,
                "filename": uploaded_file.name,
                "patient_name": structured_data.get('patient_name', 'Not Available'),
                "exam_date": structured_data.get('exam_date', 'Not Available'),
                "findings": json.dumps(structured_data),
                "report_metadata": json.dumps(metadata),
                "raw_ocr_text": combined_text  # Store raw text
            })
        except Exception as db_error:
            logging.error(f"Database save error: {str(db_error)}")
            # Continue processing even if DB save fails
        
        # Add more detailed logging
        logging.info(f"Extraction complete for {uploaded_file.name}")
        logging.info(f"Fields extracted: {list(structured_data.keys())}")
        logging.info(f"Patient name: {structured_data.get('patient_name', 'Not Available')}")
        logging.info(f"Exam date: {structured_data.get('exam_date', 'Not Available')}")
        logging.info(f"BIRADS score: {structured_data.get('birads_score', 'Not Available')}")
        
        return structured_data, metadata
        
    except Exception as e:
        logging.error(f"PDF processing error: {str(e)}", exc_info=True)
        default_output = ocr_utils.default_structured_output()
        default_output['raw_ocr_text'] = f"Error during processing: {str(e)}"
        return default_output, {"error": str(e), "filename": uploaded_file.name}

# Streamlit UI
st.title("Medical Report Processor")

uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

if uploaded_files:
    all_results = []
    
    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Add direct debug output
    st.subheader("Debug: Raw OCR Text")
    debug_expander = st.expander("Show Raw OCR Text (for debugging)")
    
    # Process each file
    for i, uploaded_file in enumerate(uploaded_files):
        status_text.text(f"Processing {uploaded_file.name}...")
        
        # Process the PDF
        result, metadata = process_pdf(uploaded_file)
        
        # Show raw text in debug expander
        with debug_expander:
            st.text_area(f"Raw text from {uploaded_file.name}", result.get('raw_ocr_text', 'No text extracted'), height=200)
        
        # Add to results
        all_results.append(result)
        
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
            'recommendation', 'testing_provider', 'ultrasound_results',
            'raw_ocr_text'  # Add raw OCR text column
        ]
        
        # Create DataFrame with consistent columns
        df = pd.DataFrame(all_results)
        
        # Add missing columns with default values
        for col in required_columns:
            if col not in df.columns:
                df[col] = "Not Available"
        
        # Replace empty values with "Not Available"
        df = df.fillna("Not Available")
        
        # Store in session state
        st.session_state.df = df
        
        # Display results with tabs
        tab1, tab2, tab3, tab4 = st.tabs(["Structured Data", "Raw OCR Text", "Debug Info", "Database"])
        
        with tab1:
            st.dataframe(df.drop(columns=['raw_ocr_text']), use_container_width=True)
        
        with tab2:
            # Show raw OCR text for selected document
            selected_doc = st.selectbox("Select document to view raw text:", df['patient_name'] + " - " + df['document_date'])
            if selected_doc:
                doc_idx = df.index[df['patient_name'] + " - " + df['document_date'] == selected_doc][0]
                st.text_area("Raw OCR Text", df.iloc[doc_idx]['raw_ocr_text'], height=400)
        
        with tab3:
            st.subheader("Extraction Debug Information")
            st.write("Field extraction success rate:")
            
            # Calculate extraction success
            extraction_stats = {}
            for col in required_columns:
                if col != 'raw_ocr_text':
                    success_rate = (df[col] != "Not Available").mean() * 100
                    extraction_stats[col] = success_rate
            
            # Display as bar chart
            stats_df = pd.DataFrame(list(extraction_stats.items()), columns=['Field', 'Success Rate (%)'])
            st.bar_chart(stats_df.set_index('Field'))

        with tab4:
            st.header("Database Management")
            
            # Show stored reports
            try:
                session = Session()
                reports = session.query(Report).all()
                
                if reports:
                    report_data = [{
                        "Filename": r.filename,
                        "Patient": r.patient_name,
                        "Exam Date": r.exam_date,
                        "Processed": r.processed_at.strftime("%Y-%m-%d %H:%M")
                    } for r in reports]
                    
                    st.dataframe(pd.DataFrame(report_data))
                    
                    # Add export option
                    if st.button("Export All Reports"):
                        export_data = []
                        for r in reports:
                            report_dict = json.loads(r.findings)
                            report_dict["filename"] = r.filename
                            report_dict["processed_at"] = r.processed_at.strftime("%Y-%m-%d %H:%M")
                            export_data.append(report_dict)
                        
                        export_df = pd.DataFrame(export_data)
                        st.download_button(
                            "Download Export",
                            export_df.to_csv(index=False),
                            "all_reports_export.csv",
                            "text/csv"
                        )
                    
                    # Add clear option
                    if st.button("Clear Database"):
                        if st.checkbox("I understand this will delete all stored reports"):
                            session.query(Report).delete()
                            session.commit()
                            st.success("Database cleared successfully")
                        else:
                            st.warning("Please confirm deletion by checking the box")
                else:
                    st.info("No reports stored in database")
            
            except Exception as e:
                st.error(f"Database error: {str(e)}")
            finally:
                if 'session' in locals():
                    session.close()

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

# Initialize Donut model
@st.cache_resource
def init_donut():
    """Initialize Donut model with enhanced validation"""
    try:
        processor = DonutProcessor.from_pretrained(
            "naver-clova-ix/donut-base-finetuned-docvqa",
            device_map="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            torch.cuda.empty_cache()
            
        model = VisionEncoderDecoderModel.from_pretrained(
            "naver-clova-ix/donut-base-finetuned-docvqa",
            device_map="cuda" if torch.cuda.is_available() else "cpu"
        )
        
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
tab1, tab2, tab3, tab4 = st.tabs(["OCR Processing", "Data Analysis", "Chatbot", "Database"])

with tab1:
    st.header("OCR Processing")
    st.write("Upload your scanned mammogram reports (PDFs) to extract data.")
    
    # PDF Uploader
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type="pdf",
        accept_multiple_files=True
    )
    
    # Add debug option
    debug_mode = st.checkbox("Enable Debug Mode", value=True, 
                            help="Show raw OCR text and extraction details")
    
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
                    result, metadata = process_pdf(file)
                    
                    # Show raw text in debug expander
                    with debug_expander:
                        st.text_area(f"Raw text from {file.name}", result.get('raw_ocr_text', 'No text extracted'), height=200)
                    
                    # Add to results
                    all_results.append(result)
                    
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
                        'recommendation', 'testing_provider', 'ultrasound_results',
                        'raw_ocr_text'  # Add raw OCR text column
                    ]
                    
                    # Create DataFrame with consistent columns
                    df = pd.DataFrame(all_results)
                    
                    # Add missing columns with default values
                    for col in required_columns:
                        if col not in df.columns:
                            df[col] = "Not Available"
                    
                    # Replace empty values with "Not Available"
                    df = df.fillna("Not Available")
                    
                    # Store in session state
                    st.session_state.df = df
                    
                    # Display results with tabs
                    tab1, tab2, tab3, tab4 = st.tabs(["Structured Data", "Raw OCR Text", "Debug Info", "Database"])
                    
                    with tab1:
                        st.dataframe(df.drop(columns=['raw_ocr_text']), use_container_width=True)
                    
                    with tab2:
                        # Show raw OCR text for selected document
                        selected_doc = st.selectbox("Select document to view raw text:", df['patient_name'] + " - " + df['document_date'])
                        if selected_doc:
                            doc_idx = df.index[df['patient_name'] + " - " + df['document_date'] == selected_doc][0]
                            st.text_area("Raw OCR Text", df.iloc[doc_idx]['raw_ocr_text'], height=400)
                    
                    with tab3:
                        st.subheader("Extraction Debug Information")
                        st.write("Field extraction success rate:")
                        
                        # Calculate extraction success
                        extraction_stats = {}
                        for col in required_columns:
                            if col != 'raw_ocr_text':
                                success_rate = (df[col] != "Not Available").mean() * 100
                                extraction_stats[col] = success_rate
                        
                        # Display as bar chart
                        stats_df = pd.DataFrame(list(extraction_stats.items()), columns=['Field', 'Success Rate (%)'])
                        st.bar_chart(stats_df.set_index('Field'))
                    with tab4:
                        st.header("Database Management")
                        
                        # Show stored reports
                        try:
                            session = Session()
                            reports = session.query(Report).all()
                            
                            if reports:
                                report_data = [{
                                    "Filename": r.filename,
                                    "Patient": r.patient_name,
                                    "Exam Date": r.exam_date,
                                    "Processed": r.processed_at.strftime("%Y-%m-%d %H:%M")
                                } for r in reports]
                                
                                st.dataframe(pd.DataFrame(report_data))
                                
                                # Add export option
                                if st.button("Export All Reports"):
                                    export_data = []
                                    for r in reports:
                                        report_dict = json.loads(r.findings)
                                        report_dict["filename"] = r.filename
                                        report_dict["processed_at"] = r.processed_at.strftime("%Y-%m-%d %H:%M")
                                        export_data.append(report_dict)
                                    
                                    export_df = pd.DataFrame(export_data)
                                    st.download_button(
                                        "Download Export",
                                        export_df.to_csv(index=False),
                                        "all_reports_export.csv",
                                        "text/csv"
                                    )
                                
                                # Add clear option
                                if st.button("Clear Database"):
                                    if st.checkbox("I understand this will delete all stored reports"):
                                        session.query(Report).delete()
                                        session.commit()
                                        st.success("Database cleared successfully")
                                    else:
                                        st.warning("Please confirm deletion by checking the box")
                            else:
                                st.info("No reports stored in database")
                        
                        except Exception as e:
                            st.error(f"Database error: {str(e)}")
                        finally:
                            if 'session' in locals():
                                session.close()
                else:
                    st.warning("No valid data extracted from PDFs")
            
            # If debug mode is enabled, show raw text immediately
            if debug_mode and uploaded_files:
                st.subheader("Raw OCR Text (Debug)")
                for file in uploaded_files:
                    with pdfplumber.open(file) as pdf:
                        for i, page in enumerate(pdf.pages):
                            page_text = page.extract_text() or "No text extracted with pdfplumber"
                            st.text_area(f"{file.name} - Page {i+1}", page_text, height=200)
    
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

    # Add this to the OCR Processing tab
    if not ocr_utils.check_tesseract_installation():
        st.error("""
        ### Tesseract OCR is not installed or not in your PATH
        
        #### Windows Installation:
        1. Download the installer from [UB-Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)
        2. Install with default options
        3. Add to PATH: `C:\\Program Files\\Tesseract-OCR`
        4. Restart the application
        
        #### Mac Installation:
        ```
        brew install tesseract
        ```
        
        #### Linux Installation:
        ```
        sudo apt-get install tesseract-ocr
        ```
        """)

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

# Add to sidebar section at bottom of file
with st.sidebar:
    st.title("Mammogram Analysis Dashboard")
    
    # Theme toggle explanation
    st.write("💡 **Tip:** Change between light/dark mode using the ⋮ menu in the top-right corner.")
    
    # Add cache clearing button
    if st.button("Clear All Caches"):
        if clear_all_caches():
            st.success("All caches cleared successfully!")
        else:
            st.error("Failed to clear caches")
    
    # Rest of your sidebar content
    st.markdown("""
    **Clinical Decision Support System**  
    AI-powered analysis of mammogram reports with:
    - PDF OCR extraction
    - Medical NLP processing
    - BIRADS classification
    - Clinical insights generation
    """)

    # Add database status
    try:
        session = Session()
        report_count = session.query(Report).count()
        st.success(f"Database connected: {report_count} reports stored")
    except Exception as e:
        st.error(f"Database error: {str(e)}")
    finally:
        if 'session' in locals():
            session.close()

@st.cache_resource
def load_translation_model():
    return pipeline(
        "translation_fr_to_en",
        model="Helsinki-NLP/opus-mt-fr-en",
        device=0 if torch.cuda.is_available() else -1
    )

# Initialize database at startup
try:
    init_db()
    logging.info("Database initialized successfully")
except Exception as e:
    logging.error(f"Database initialization error: {str(e)}")

# 4. Add error handling for missing nltk data
def ensure_nltk_resources():
    """Ensure NLTK resources are downloaded"""
    import nltk
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

# Call this function early in the app
ensure_nltk_resources()

if __name__ == "__main__":
    st.write("Medical AI Assistant is running...") 
