import os
import io
import tempfile
import logging
import numpy as np
import pandas as pd
import streamlit as st
from io import BytesIO
from typing import Dict, List, Tuple, Optional, BinaryIO, Any
from datetime import datetime
import base64
import re

# Try importing OCR dependencies
try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False
    logging.warning("PaddleOCR not available. OCR features will be limited.")

# Try importing PyMuPDF
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except (ImportError, RuntimeError):
    PYMUPDF_AVAILABLE = False
    logging.warning("PyMuPDF not available. PDF processing will be limited.")

# Try importing transformers for document understanding
try:
    import torch
    from transformers import DonutProcessor, VisionEncoderDecoderModel
    from PIL import Image
    DONUT_AVAILABLE = True
except ImportError:
    DONUT_AVAILABLE = False
    logging.warning("Transformers or PyTorch not available. Advanced document understanding will be limited.")

# Set up logging
logger = logging.getLogger(__name__)

@st.cache_resource(show_spinner="Initializing OCR Engine...")
def init_paddle():
    """Initialize PaddleOCR engine with caching"""
    try:
        # Initialize PaddleOCR with English and optimization
        ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False, show_log=False)
        return ocr
    except Exception as e:
        logger.error(f"Failed to initialize PaddleOCR: {e}")
        return None

@st.cache_resource
def init_donut():
    """Initialize Donut document understanding model with caching"""
    try:
        processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
        model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
        
        # Enable evaluation mode for inference
        model.eval()
        
        # Move to GPU if available
        if torch.cuda.is_available():
            model.to("cuda")
        
        return {"processor": processor, "model": model}
    except Exception as e:
        logger.error(f"Failed to initialize Donut model: {e}")
        return None

def process_pdf(uploaded_file):
    """
    Process an uploaded PDF file to extract text, apply OCR, and analyze content

    Args:
        uploaded_file: Streamlit uploaded file object

    Returns:
        Dict containing processed data including text, metadata, and analysis
    """
    if uploaded_file is None:
        return None

    # Initialize the status indicators
    status = {
        "ocr_engine": "Initializing...",
        "text_extraction": "Waiting...",
        "page_count": 0,
        "current_page": 0,
        "analysis": "Waiting..."
    }

    # Define a placeholder for progress display
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text("Initializing PDF processing...")

    try:
        # Initialize OCR if available
        ocr_engine = None
        donut_model = None
        
        if PADDLE_AVAILABLE:
            status["ocr_engine"] = "Loading PaddleOCR..."
            status_text.text("Loading OCR engine...")
            ocr_engine = init_paddle()
            status["ocr_engine"] = "Ready" if ocr_engine else "Failed to load"
        else:
            status["ocr_engine"] = "Not available"
        
        if DONUT_AVAILABLE:
            status["donut"] = "Loading Donut model..."
            status_text.text("Loading document understanding model...")
            donut_model = init_donut()
            status["donut"] = "Ready" if donut_model else "Failed to load"
        else:
            status["donut"] = "Not available"

        # Read file content
        file_content = uploaded_file.read()
        filename = uploaded_file.name

        # Process with PyMuPDF if available
        results = {
            "filename": filename,
            "text": "",
            "metadata": {},
            "pages_text": [],
            "ocr_text": "",
            "page_images": [],
            "pdf_preview": None,
            "structured_data": None
        }

        if PYMUPDF_AVAILABLE:
            # Use PyMuPDF to extract text and render page images
            status["text_extraction"] = "Extracting with PyMuPDF..."
            status_text.text("Extracting text from PDF...")
            
            # Create a temporary file for PyMuPDF to read from
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                temp_file.write(file_content)
                temp_file_path = temp_file.name
            
            # Open the PDF with PyMuPDF
            doc = fitz.open(temp_file_path)
            status["page_count"] = len(doc)
            
            # Extract text and render page images
            text_content = ""
            pages_text = []
            page_images = []
            
            for page_num, page in enumerate(doc):
                status["current_page"] = page_num + 1
                status_text.text(f"Processing page {page_num + 1} of {len(doc)}...")
                progress_bar.progress((page_num + 1) / len(doc))
                
                # Extract text from this page
                page_text = page.get_text()
                text_content += page_text
                pages_text.append(page_text)
                
                # Render the page to an image
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                image_bytes = pix.tobytes("png")
                page_images.append(image_bytes)

                # If OCR is available, apply it to the current page image
                if ocr_engine:
                    try:
                        # Convert the image bytes to a numpy array for OCR
                        image = Image.open(io.BytesIO(image_bytes))
                        ocr_result = ocr_engine.ocr(np.array(image), cls=True)
                        
                        # Extract text from OCR result
                        for idx, result in enumerate(ocr_result):
                            if result:
                                for line in result:
                                    if len(line) >= 2 and isinstance(line[1], tuple) and len(line[1]) >= 1:
                                        results["ocr_text"] += line[1][0] + " "
                    except Exception as e:
                        logger.error(f"OCR failed on page {page_num + 1}: {e}")
                
                # If Donut is available, apply it to the current page image for structured extraction
                if donut_model and page_num == 0:  # Apply to first page only for efficiency
                    try:
                        processor = donut_model["processor"]
                        model = donut_model["model"]
                        
                        # Load the image
                        image = Image.open(io.BytesIO(image_bytes))
                        
                        # Prepare inputs for the model
                        pixel_values = processor(image, return_tensors="pt").pixel_values
                        if torch.cuda.is_available():
                            pixel_values = pixel_values.to("cuda")
                        
                        # Generate predictions
                        task_prompt = "<s_cord-v2>"
                        decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids
                        if torch.cuda.is_available():
                            decoder_input_ids = decoder_input_ids.to("cuda")
                        
                        outputs = model.generate(
                            pixel_values,
                            decoder_input_ids=decoder_input_ids,
                            max_length=model.decoder.config.max_position_embeddings,
                            early_stopping=True,
                            pad_token_id=processor.tokenizer.pad_token_id,
                            eos_token_id=processor.tokenizer.eos_token_id,
                            use_cache=True,
                            num_beams=1,
                            bad_words_ids=[[processor.tokenizer.unk_token_id]],
                            return_dict_in_generate=True,
                        )
                        
                        # Process the outputs
                        sequence = processor.batch_decode(outputs.sequences)[0]
                        sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
                        sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  # Remove the first task start token
                        
                        # Save the structured data
                        results["structured_data_raw"] = sequence
                        
                        # Try to parse into a better structure
                        try:
                            import json
                            parsed = processor.token2json(sequence)
                            results["structured_data"] = parsed
                        except Exception as e:
                            logger.error(f"Failed to parse structured data: {e}")
                    except Exception as e:
                        logger.error(f"Donut processing failed: {e}")
            
            # Clean up the temporary file
            os.unlink(temp_file_path)
            
            # Store the extracted text and images
            results["text"] = text_content
            results["pages_text"] = pages_text
            results["page_images"] = page_images
            
            # Extract metadata if available
            if hasattr(doc, 'metadata'):
                metadata = doc.metadata
                results["metadata"] = {
                    "title": metadata.get("title", ""),
                    "author": metadata.get("author", ""),
                    "subject": metadata.get("subject", ""),
                    "creator": metadata.get("creator", ""),
                    "producer": metadata.get("producer", "")
                }
            
            # Extract a preview for display
            if page_images and len(page_images) > 0:
                results["pdf_preview"] = page_images[0]
        else:
            # Fallback if PyMuPDF is not available - use basic OCR if available
            status["text_extraction"] = "PyMuPDF not available, using fallback..."
            
            if ocr_engine:
                status_text.text("Using OCR to extract text...")
                
                try:
                    # Convert PDF to images first since we don't have PyMuPDF
                    try:
                        from pdf2image import convert_from_bytes
                        images = convert_from_bytes(file_content)
                        
                        status["page_count"] = len(images)
                        text_content = ""
                        
                        for page_num, image in enumerate(images):
                            status["current_page"] = page_num + 1
                            status_text.text(f"OCR processing page {page_num + 1} of {len(images)}...")
                            progress_bar.progress((page_num + 1) / len(images))
                            
                            # Convert PIL image to bytes for display
                            img_byte_arr = BytesIO()
                            image.save(img_byte_arr, format='PNG')
                            page_images.append(img_byte_arr.getvalue())
                            
                            # Apply OCR to the image
                            ocr_result = ocr_engine.ocr(np.array(image), cls=True)
                            
                            # Extract text from OCR result
                            page_text = ""
                            for idx, result in enumerate(ocr_result):
                                if result:
                                    for line in result:
                                        if len(line) >= 2 and isinstance(line[1], tuple) and len(line[1]) >= 1:
                                            page_text += line[1][0] + " "
                            
                            text_content += page_text
                            pages_text.append(page_text)
                        
                        results["text"] = text_content
                        results["pages_text"] = pages_text
                        results["page_images"] = page_images
                        results["ocr_text"] = text_content
                        
                        # Extract a preview for display
                        if page_images and len(page_images) > 0:
                            results["pdf_preview"] = page_images[0]
                    except ImportError:
                        status_text.text("pdf2image not available, OCR processing limited...")
                        logger.warning("pdf2image not available, cannot convert PDF to images for OCR")
                except Exception as e:
                    status_text.text(f"OCR processing failed: {str(e)}")
                    logger.error(f"OCR processing failed: {e}")
            else:
                status_text.text("Both PyMuPDF and OCR are unavailable. Cannot process PDF.")
                logger.warning("Both PyMuPDF and OCR are unavailable. Cannot process PDF.")

        # Analyze the extracted text
        status["analysis"] = "Analyzing content..."
        status_text.text("Analyzing extracted text...")
        
        # Create structured data frame from extracted text
        results["dataframe"] = create_report_dataframe(results["text"])
        
        # Complete
        progress_bar.progress(100)
        status_text.text("Processing complete!")
        progress_bar.empty()
        status_text.empty()
        
        return results
    
    except Exception as e:
        logger.error(f"PDF processing failed: {e}")
        status_text.error(f"Error: {str(e)}")
        progress_bar.empty()
        return {
            "filename": uploaded_file.name if uploaded_file else "unknown",
            "error": str(e),
            "text": "Processing failed"
        }

def process_pdf_batch(batch):
    """Process a batch of PDF files and return combined results"""
    results = []
    
    for uploaded_file in batch:
        try:
            # Process the PDF
            pdf_data = process_pdf(uploaded_file)
            if pdf_data and not pdf_data.get("error"):
                results.append(pdf_data)
        except Exception as e:
            logger.error(f"Error processing {uploaded_file.name}: {e}")
            
    # Create a combined dataframe
    combined_df = None
    if results:
        dfs = [item.get("dataframe") for item in results if "dataframe" in item]
        if dfs:
            combined_df = pd.concat(dfs, ignore_index=True)
            
    return {"individual_results": results, "combined_dataframe": combined_df}

def process_page(page):
    """Process a single page image with OCR"""
    if not PADDLE_AVAILABLE:
        return "OCR engine not available"
    
    try:
        ocr_engine = init_paddle()
        result = ocr_engine.ocr(np.array(page), cls=True)
        
        # Extract text from result
        text = ""
        for idx, line in enumerate(result):
            if line:
                for item in line:
                    if len(item) >= 2 and isinstance(item[1], tuple) and len(item[1]) >= 1:
                        text += item[1][0] + " "
        
        return text
    except Exception as e:
        logger.error(f"Page OCR failed: {e}")
        return f"OCR failed: {str(e)}"

def extract_date_from_text(text):
    """Extract date from text"""
    if not text:
        return ""
    
    # Try to find an exam date
    date_pattern = r'(?:EXAM DATE|DATE OF EXAM|EXAMINATION DATE)[:\s]+([^\n]+)'
    date_match = re.search(date_pattern, text, re.IGNORECASE)
    if date_match:
        date_text = date_match.group(1).strip()
        
        # Try to parse various date formats
        try:
            # Try ISO format (2023-01-15)
            if re.search(r'\d{4}-\d{2}-\d{2}', date_text):
                return re.search(r'\d{4}-\d{2}-\d{2}', date_text).group(0)
            
            # Try MM/DD/YYYY format
            elif re.search(r'\d{1,2}/\d{1,2}/\d{4}', date_text):
                mm_dd_yyyy = re.search(r'(\d{1,2})/(\d{1,2})/(\d{4})', date_text)
                month, day, year = mm_dd_yyyy.groups()
                return f"{year}-{int(month):02d}-{int(day):02d}"
            
            # Try Month DD, YYYY format
            elif re.search(r'[A-Za-z]{3,9}\s+\d{1,2},\s+\d{4}', date_text):
                month_names = {
                    'jan': '01', 'feb': '02', 'mar': '03', 'apr': '04', 
                    'may': '05', 'jun': '06', 'jul': '07', 'aug': '08', 
                    'sep': '09', 'oct': '10', 'nov': '11', 'dec': '12'
                }
                match = re.search(r'([A-Za-z]{3,9})\s+(\d{1,2}),\s+(\d{4})', date_text)
                month, day, year = match.groups()
                month_num = month_names.get(month.lower()[:3], '01')
                return f"{year}-{month_num}-{int(day):02d}"
            
            # Just return the original date text if we can't parse it
            return date_text
        except:
            return date_text
    
    return ""

def extract_birads_number(value):
    """Extract the BIRADS score from text that might contain it"""
    if not value:
        return None
    
    # Convert to string if needed
    if not isinstance(value, str):
        value = str(value)
    
    value = value.upper()
    
    # Direct number match
    if re.match(r'^[0-6]$', value):
        return value
    
    # Format: "BIRADS: X" or "BI-RADS X" etc.
    birads_pattern = r'(?:BIRADS|BI-RADS|BIRAD|BI-RAD)[^0-6]*([0-6])'
    match = re.search(birads_pattern, value, re.IGNORECASE)
    if match:
        return match.group(1)
    
    # Format with "Category X"
    category_pattern = r'CATEGORY\s*([0-6])'
    match = re.search(category_pattern, value, re.IGNORECASE)
    if match:
        return match.group(1)
    
    # Look for any digit that might be a BIRADS score
    digit_match = re.search(r'[0-6]', value)
    if digit_match:
        return digit_match.group(0)
    
    return None

def standardize_birads(df):
    """Standardize BIRADS values in the dataframe"""
    if df is None or df.empty:
        return df
    
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    
    # Check if BIRADS column exists
    if 'BIRADS Score' in result_df.columns:
        # Apply standardization to non-null values
        mask = result_df['BIRADS Score'].notna()
        result_df.loc[mask, 'BIRADS Score'] = result_df.loc[mask, 'BIRADS Score'].astype(str).apply(extract_birads_number)
    
    return result_df

def extract_exam_type(text):
    """Extract examination type from text"""
    if not text:
        return None
    
    # Common examination types
    exam_types = {
        "MAMMOGRAM": ["MAMMOGRAM", "MAMMO", "MAMMOGRAPHY"],
        "ULTRASOUND": ["ULTRASOUND", "US", "SONOGRAM", "SONO"],
        "MRI": ["MRI", "MAGNETIC RESONANCE", "MAGNETIC RESONANCE IMAGING"],
        "CT SCAN": ["CT SCAN", "CT", "COMPUTED TOMOGRAPHY"],
        "X-RAY": ["X-RAY", "XRAY", "RADIOGRAPH"],
        "PET SCAN": ["PET", "POSITRON EMISSION", "PET SCAN", "PET-CT"]
    }
    
    text_upper = text.upper()
    
    for exam_type, keywords in exam_types.items():
        for keyword in keywords:
            if re.search(r'\b' + re.escape(keyword) + r'\b', text_upper):
                return exam_type
    
    return None

def extract_sections(text):
    """Extract standard report sections"""
    if not text:
        return {}
    
    sections = {}
    
    # Common section headers
    section_patterns = {
        "clinical_history": r'(?:CLINICAL HISTORY|HISTORY|CLINICAL DATA|INDICATION):\s*(.*?)(?=\n\s*(?:[A-Z][A-Z\s]+:|\Z))',
        "findings": r'(?:FINDINGS|FINDING):\s*(.*?)(?=\n\s*(?:[A-Z][A-Z\s]+:|\Z))',
        "impression": r'(?:IMPRESSION|ASSESSMENT|CONCLUSION):\s*(.*?)(?=\n\s*(?:[A-Z][A-Z\s]+:|\Z))',
        "recommendation": r'(?:RECOMMENDATION|RECOMMENDATIONS|ADVISED):\s*(.*?)(?=\n\s*(?:[A-Z][A-Z\s]+:|\Z))'
    }
    
    for section_name, pattern in section_patterns.items():
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            sections[section_name] = match.group(1).strip()
    
    return sections

def extract_patient_info(text):
    """Extract patient information from text"""
    if not text:
        return {}
    
    patient_info = {}
    
    # Extract patient name
    name_pattern = r'(?:PATIENT NAME|NAME)[:\s]+([^\n]+)'
    name_match = re.search(name_pattern, text, re.IGNORECASE)
    if name_match:
        patient_info['name'] = name_match.group(1).strip()
    
    # Extract patient ID
    id_pattern = r'(?:PATIENT ID|MRN|MEDICAL RECORD NUMBER)[:\s]+([^\n]+)'
    id_match = re.search(id_pattern, text, re.IGNORECASE)
    if id_match:
        patient_info['id'] = id_match.group(1).strip()
    
    # Extract patient age
    age_pattern = r'(?:PATIENT AGE|AGE)[:\s]+([^\n]+)'
    age_match = re.search(age_pattern, text, re.IGNORECASE)
    if age_match:
        patient_info['age'] = age_match.group(1).strip()
    
    # Extract patient gender
    gender_pattern = r'(?:PATIENT GENDER|GENDER|SEX)[:\s]+([^\n]+)'
    gender_match = re.search(gender_pattern, text, re.IGNORECASE)
    if gender_match:
        patient_info['gender'] = gender_match.group(1).strip()
    
    return patient_info

def create_report_dataframe(text):
    """
    Create a structured dataframe from extracted text
    
    Args:
        text: Extracted text from PDF
        
    Returns:
        DataFrame with structured report data
    """
    # Create a dictionary for the dataframe
    data = {
        "Document Date": None,
        "Examination Date": None,
        "Examination Type": None,
        "BIRADS Score": None,
        "Patient Name": None,
        "Patient ID": None,
        "Patient Age": None,
        "Patient Gender": None,
        "Findings": None,
        "Impression": None
    }
    
    if not text:
        return pd.DataFrame([data])
    
    # Extract date information
    date_pattern = r'(?:EXAM DATE|DATE OF EXAM|EXAMINATION DATE):\s*(.{10,30})'
    date_match = re.search(date_pattern, text, re.IGNORECASE)
    if date_match:
        date_text = date_match.group(1).strip()
        extracted_date = extract_date_from_text(date_text)
        if extracted_date:
            data["Examination Date"] = extracted_date
    
    # Try to find a document date
    doc_date_pattern = r'(?:REPORT DATE|DATE OF REPORT|REPORTED ON):\s*(.{10,30})'
    doc_date_match = re.search(doc_date_pattern, text, re.IGNORECASE)
    if doc_date_match:
        doc_date_text = doc_date_match.group(1).strip()
        extracted_doc_date = extract_date_from_text(doc_date_text)
        if extracted_doc_date:
            data["Document Date"] = extracted_doc_date
    
    # Extract examination type
    data["Examination Type"] = extract_exam_type(text)
    
    # Extract BIRADS score
    birads_pattern = r'(?:BIRADS|BI-RADS)[^0-6]*([0-6])'
    birads_match = re.search(birads_pattern, text, re.IGNORECASE)
    if birads_match:
        data["BIRADS Score"] = birads_match.group(1)
    
    # Extract sections
    sections = extract_sections(text)
    if "findings" in sections:
        data["Findings"] = sections["findings"]
    if "impression" in sections:
        data["Impression"] = sections["impression"]
    
    # Extract patient info
    patient_info = extract_patient_info(text)
    if "name" in patient_info:
        data["Patient Name"] = patient_info["name"]
    if "id" in patient_info:
        data["Patient ID"] = patient_info["id"]
    if "age" in patient_info:
        data["Patient Age"] = patient_info["age"]
    if "gender" in patient_info:
        data["Patient Gender"] = patient_info["gender"]
    
    # Create DataFrame
    df = pd.DataFrame([data])
    
    return df 