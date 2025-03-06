import os
import tempfile
import logging
import re
from datetime import datetime
from typing import Dict, Any, List, Optional, BinaryIO, Tuple
import pandas as pd
import numpy as np
import io
import streamlit as st
import json

# Try importing PyMuPDF, but provide fallback if it fails
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except (ImportError, RuntimeError) as e:
    PYMUPDF_AVAILABLE = False
    logging.warning(f"PyMuPDF (fitz) not available: {e}. Will use alternative PDF processing.")

# Try importing alternative PDF library
try:
    from pdf2image import convert_from_bytes
    from pytesseract import image_to_string
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False
    logging.warning("pdf2image not available. PDF text extraction may be limited.")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logging.warning("PIL not available. Image processing will be limited.")

from models.model_loader import get_nlp_model
from database.operations import save_report
from api.claude_client import claude_client

logger = logging.getLogger(__name__)

def process_document(file_obj: BinaryIO, models: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a PDF document and extract structured information using available models
    
    Args:
        file_obj: File-like object containing the PDF
        models: Dictionary of loaded models
        
    Returns:
        Dictionary containing extracted information and processing results
    """
    logger.info("Starting document processing")
    
    # Create a temporary file for the PDF
    file_content = file_obj.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
        temp_file.write(file_content)
        temp_file_path = temp_file.name
    
    try:
        # Extract text and images from PDF
        doc_text, page_images = extract_from_pdf(temp_file_path, file_content)
        
        # Initialize the results dictionary
        results = {
            'document_text': doc_text,
            'page_images': page_images,
            'timestamp': datetime.now().isoformat(),
            'filename': getattr(file_obj, 'name', 'unnamed_document.pdf'),
        }
        
        # Extract structured data using NLP models
        extracted_data = extract_structured_data(doc_text, models)
        
        # Extract detailed medical report structure
        report_structure = extract_medical_report_structure(doc_text)
        
        # Merge the structured data with report structure
        if report_structure:
            if 'sections' in report_structure:
                extracted_data['sections'] = report_structure['sections']
            if 'metadata' in report_structure:
                extracted_data['metadata'] = report_structure['metadata']
            
            # Add BIRADS score if available
            if 'birads_score' in report_structure:
                extracted_data['birads_score'] = report_structure['birads_score']
                
            # Add patient info if available
            if 'patient_info' in report_structure:
                extracted_data['patient_info'].update(report_structure['patient_info'])
        
        # Create a structured dataframe for display
        dataframe_rows = create_structured_dataframe(extracted_data, results['filename'])
        extracted_data['structured_data'] = dataframe_rows
        
        results['extracted_data'] = extracted_data
        
        # Save to database
        report_id = save_report(
            filename=results['filename'],
            text_content=doc_text,
            processed_data=results
        )
        results['report_id'] = report_id
        
        logger.info(f"Document processed successfully, saved with ID: {report_id}")
        return results
    
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        return None
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

def extract_from_pdf(pdf_path: str, pdf_content: bytes = None) -> Tuple[str, List[Any]]:
    """
    Extract text and images from a PDF file with fallbacks for different libraries
    
    Args:
        pdf_path: Path to the PDF file
        pdf_content: Raw PDF content as bytes (optional)
        
    Returns:
        Tuple of (text_content, list_of_images)
    """
    logger.info(f"Extracting content from PDF: {pdf_path}")
    
    text_content = ""
    page_images = []
    
    # Try PyMuPDF first if available
    if PYMUPDF_AVAILABLE:
        try:
            # Open the PDF with PyMuPDF
            doc = fitz.open(pdf_path)
            
            # Extract text
            for page_num, page in enumerate(doc):
                text_content += page.get_text()
                
                # Try to extract images 
                try:
                    # Extract images from page
                    image_list = page.get_images(full=True)
                    
                    for img_index, img_info in enumerate(image_list):
                        try:
                            xref = img_info[0]
                            base_image = doc.extract_image(xref)
                            image_bytes = base_image["image"]
                            
                            # Convert to PIL Image
                            if PIL_AVAILABLE:
                                image = Image.open(io.BytesIO(image_bytes))
                                page_images.append(image)
                        except Exception as e:
                            logger.warning(f"Could not process image {img_index} on page {page_num}: {e}")
                except Exception as e:
                    logger.warning(f"Error extracting images from page {page_num}: {e}")
            
            return text_content, page_images
        
        except Exception as e:
            logger.warning(f"PyMuPDF extraction failed: {e}. Trying alternative method.")
    
    # If PyMuPDF failed or isn't available, try pdf2image + pytesseract
    if PDF2IMAGE_AVAILABLE and PIL_AVAILABLE:
        try:
            logger.info("Using pdf2image for PDF extraction")
            
            # Convert PDF to images
            if pdf_content:
                images = convert_from_bytes(pdf_content)
            else:
                with open(pdf_path, 'rb') as f:
                    images = convert_from_bytes(f.read())
            
            # Store page images and extract text with OCR
            for i, img in enumerate(images):
                # Store the page image
                page_images.append(img)
                
                # Extract text with OCR
                try:
                    page_text = image_to_string(img)
                    text_content += page_text + "\n\n"
                except Exception as e:
                    logger.warning(f"OCR failed for page {i}: {e}")
            
            return text_content, page_images
            
        except Exception as e:
            logger.warning(f"pdf2image extraction failed: {e}")
    
    # Last resort - if no PDF extraction method worked, return empty results
    logger.warning("All PDF extraction methods failed. Returning empty results.")
    return text_content, page_images

def extract_structured_data(text_content: str, models: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract structured data from text content using NLP models
    
    Args:
        text_content: Text content from the document
        models: Dictionary of loaded models
        
    Returns:
        Dictionary containing structured extracted data
    """
    logger.info("Extracting structured data from text")
    
    # Initialize result structure
    extracted_data = {
        'medical_terms': [],
        'conditions': [],
        'measurements': {},
        'definitions': {},
        'patient_info': {},
        'diagnoses': [],
        'treatments': [],
        'medications': [],
        'birads_score': None,
        'examination_date': None,
        'document_date': None,
        'examination_type': None,
        'findings': "",
        'impression': "",
    }
    
    # Use Claude if available for high-level structured extraction
    if claude_client:
        try:
            claude_analysis = extract_with_claude(text_content)
            if claude_analysis:
                # Merge Claude's analysis with our structure
                for key, value in claude_analysis.items():
                    if key in extracted_data:
                        if isinstance(value, list):
                            extracted_data[key].extend(value)
                        elif isinstance(value, dict):
                            extracted_data[key].update(value)
                        else:
                            extracted_data[key] = value
                logger.info("Successfully enhanced extraction with Claude")
        except Exception as e:
            logger.warning(f"Claude extraction failed: {e}")
    
    # Use RadBERT model for medical term extraction if available
    nlp_model = models.get('radbert')
    if nlp_model:
        try:
            # Extract medical entities
            medical_entities = extract_medical_entities(text_content, nlp_model)
            
            # Process and categorize the entities
            for entity, label, confidence in medical_entities:
                if label == "MEDICAL_CONDITION":
                    if entity not in extracted_data['conditions']:
                        extracted_data['conditions'].append(entity)
                elif label == "ANATOMY":
                    if entity not in extracted_data['medical_terms']:
                        extracted_data['medical_terms'].append(entity)
                elif label == "MEDICATION":
                    if entity not in extracted_data['medications']:
                        extracted_data['medications'].append(entity)
                elif label == "PROCEDURE":
                    if entity not in extracted_data['treatments']:
                        extracted_data['treatments'].append(entity)
        except Exception as e:
            logger.warning(f"NLP extraction error: {e}")
    
    # Extract measurements (simple regex approach)
    measurements = extract_measurements(text_content)
    if measurements:
        extracted_data['measurements'] = measurements
    
    # Extract BIRADS score if not already extracted
    if not extracted_data['birads_score']:
        birads = extract_birads_score(text_content)
        if birads:
            extracted_data['birads_score'] = birads
    
    # Extract dates if not already extracted
    if not extracted_data['examination_date'] or not extracted_data['document_date']:
        dates = extract_dates(text_content)
        if 'examination_date' in dates and not extracted_data['examination_date']:
            extracted_data['examination_date'] = dates['examination_date']
        if 'document_date' in dates and not extracted_data['document_date']:
            extracted_data['document_date'] = dates['document_date']
    
    # Prepare dataframe-ready format for display
    dataframe_data = []
    
    # Add terms with their categories
    for category in ["medical_terms", "conditions", "medications", "treatments"]:
        for item in extracted_data[category]:
            dataframe_data.append({
                "category": category.replace("_", " ").title(),
                "term": item,
                "confidence": 0.9  # Placeholder confidence value
            })
    
    # Add measurements in a similar format
    for key, value in extracted_data.get('measurements', {}).items():
        dataframe_data.append({
            "category": "Measurement",
            "term": key,
            "value": str(value),
            "confidence": 1.0
        })
    
    # Store the DataFrame-ready data if we have entries
    if dataframe_data:
        extracted_data['dataframe'] = dataframe_data
    
    return extracted_data

def extract_medical_entities(text: str, nlp_model) -> List[Tuple[str, str, float]]:
    """
    Extract medical entities from text using NLP model
    
    Args:
        text: Text to extract entities from
        nlp_model: Loaded NLP model
        
    Returns:
        List of tuples (entity_text, entity_label, confidence)
    """
    # Simple approach: split text into manageable chunks to avoid context length issues
    max_chunk_size = 512  # Adjust based on model requirements
    chunks = [text[i:i+max_chunk_size] for i in range(0, len(text), max_chunk_size)]
    
    entities = []
    for chunk in chunks:
        try:
            # Process chunk with NLP model
            doc = nlp_model(chunk)
            
            # Extract entities from the processed chunk
            for ent in doc.ents:
                # Add entity with its label and a confidence score
                # Note: Some models may provide confidence, others may not
                confidence = getattr(ent, 'confidence', 0.8)
                entities.append((ent.text, ent.label_, confidence))
        except Exception as e:
            logger.warning(f"Error processing text chunk with NLP model: {e}")
    
    return entities

def extract_measurements(text: str) -> Dict[str, Any]:
    """
    Extract common medical measurements using regex patterns
    
    Args:
        text: Medical text
        
    Returns:
        Dictionary of extracted measurements
    """
    import re
    
    measurements = {}
    
    # Extract blood pressure measurements (e.g., 120/80 mmHg)
    bp_pattern = r'(\d{2,3})[/](\d{2,3})\s*(mm ?Hg|mmHg)'
    bp_matches = re.findall(bp_pattern, text)
    if bp_matches:
        systolic, diastolic, _ = bp_matches[0]
        measurements['blood_pressure'] = f"{systolic}/{diastolic}"
        measurements['systolic'] = int(systolic)
        measurements['diastolic'] = int(diastolic)
    
    # Extract heart rate/pulse measurements
    hr_pattern = r'(?:heart rate|pulse|HR)[:\s]+(\d{2,3})'
    hr_matches = re.findall(hr_pattern, text, re.IGNORECASE)
    if hr_matches:
        measurements['heart_rate'] = int(hr_matches[0])
    
    # Extract temperature measurements
    temp_pattern = r'(?:temperature|temp)[:\s]+(\d{2,3}(?:\.\d)?)\s*(?:°C|C|°F|F)'
    temp_matches = re.findall(temp_pattern, text, re.IGNORECASE)
    if temp_matches:
        measurements['temperature'] = float(temp_matches[0])
    
    # Extract height
    height_pattern = r'(?:height)[:\s]+(\d{2,3}(?:\.\d)?)\s*(?:cm|m)'
    height_matches = re.findall(height_pattern, text, re.IGNORECASE)
    if height_matches:
        measurements['height'] = float(height_matches[0])
    
    # Extract weight
    weight_pattern = r'(?:weight)[:\s]+(\d{2,3}(?:\.\d)?)\s*(?:kg|lbs)'
    weight_matches = re.findall(weight_pattern, text, re.IGNORECASE)
    if weight_matches:
        measurements['weight'] = float(weight_matches[0])
    
    return measurements

def extract_with_claude(text: str) -> Dict[str, Any]:
    """
    Use Claude to extract structured information from medical text
    
    Args:
        text: Medical text content
        
    Returns:
        Dictionary with structured extraction results
    """
    if not claude_client:
        return {}
    
    system_prompt = """
    You are a medical data extraction assistant specialized in analyzing medical reports.
    Extract structured information from the provided medical text. Focus on:
    
    1. Patient information (age, gender, name, ID, DOB)
    2. Medical conditions/diagnoses 
    3. Medications mentioned
    4. Medical measurements and lab values
    5. Treatments or procedures
    6. Important medical terms and their definitions
    7. BIRADS score if present (for mammograms)
    8. Dates (document date, examination date)
    9. Examination type
    10. Findings and impression sections
    
    Format your response as a valid JSON object with these keys:
    - patient_info (object with name, age, gender, id, dob)
    - conditions (array of strings)
    - medications (array of strings)
    - measurements (object with key-value pairs)
    - treatments (array of strings)
    - medical_terms (array of strings)
    - definitions (object with terms as keys and definitions as values)
    - birads_score (string or number, null if not found)
    - examination_date (string in ISO format, null if not found)
    - document_date (string in ISO format, null if not found)
    - examination_type (string, e.g. "Mammogram", "MRI", null if not found)
    - findings (string containing the findings section)
    - impression (string containing the impression/conclusion section)
    
    Include only information explicitly mentioned in the text. Return only the JSON, nothing else.
    """
    
    # Truncate text if needed
    max_text_length = 10000  # Adjust based on Claude version's limit
    if len(text) > max_text_length:
        text = text[:max_text_length]
    
    try:
        # Call Claude API
        # Get the selected model from session state with a fallback
        selected_model = "claude-3-haiku-20240307"
        if 'selected_claude_model' in st.session_state:
            selected_model = st.session_state.selected_claude_model
            
        message = claude_client.messages.create(
            model=selected_model,
            system=system_prompt,
            max_tokens=4000,
            messages=[
                {"role": "user", "content": text}
            ]
        )
        
        # Parse response
        import json
        try:
            # Get content from the message
            response_text = message.content[0].text
            # Extract JSON from the response
            analysis = json.loads(response_text)
            return analysis
        except Exception as e:
            logger.warning(f"Error parsing Claude's JSON response: {e}")
            return {}
            
    except Exception as e:
        logger.warning(f"Error calling Claude API: {e}")
        return {}

def extract_medical_report_structure(text: str) -> Dict[str, Any]:
    """
    Extract structured sections from a medical report
    
    Args:
        text: The medical report text
        
    Returns:
        Dictionary with structured sections and metadata
    """
    # Common section headers in medical reports
    section_patterns = {
        'clinical_history': r'(?:CLINICAL|CLINICAL HISTORY|HISTORY|INDICATION)[\s:]+',
        'technique': r'(?:TECHNIQUE|PROCEDURE|PROTOCOL)[\s:]+',
        'findings': r'(?:FINDINGS|REPORT|RESULT|OBSERVATION)[\s:]+',
        'impression': r'(?:IMPRESSION|CONCLUSION|ASSESSMENT|DIAGNOSIS|SUMMARY)[\s:]+',
        'recommendation': r'(?:RECOMMENDATION|PLAN|FOLLOW-UP|ADVISED)[\s:]+',
    }
    
    # Initialize result structure
    result = {
        'sections': {},
        'metadata': {},
        'patient_info': {}
    }
    
    # Extract sections
    for section_name, pattern in section_patterns.items():
        section_matches = re.search(f"{pattern}(.*?)(?:(?:{list(section_patterns.values())})|$)", 
                                   text, 
                                   re.IGNORECASE | re.DOTALL)
        if section_matches:
            section_text = section_matches.group(1).strip()
            result['sections'][section_name] = section_text
    
    # Extract BIRADS score
    birads_pattern = r'(?:BIRADS|BI-RADS)[:\s]*([0-6])'
    birads_matches = re.search(birads_pattern, text, re.IGNORECASE)
    if birads_matches:
        result['birads_score'] = birads_matches.group(1)
    
    # Extract patient information patterns
    patient_name_pattern = r'(?:PATIENT|NAME)[:\s]+([A-Za-z\s]+)'
    patient_id_pattern = r'(?:PATIENT ID|ID|MRN)[:\s]+([A-Za-z0-9\-]+)'
    patient_dob_pattern = r'(?:DOB|DATE OF BIRTH)[:\s]+([0-9/\-\.]+)'
    patient_age_pattern = r'(?:AGE)[:\s]+(\d+)'
    patient_gender_pattern = r'(?:(?:SEX|GENDER)[:\s]+)(MALE|FEMALE|M|F)'
    
    # Extract patient info
    patient_name_match = re.search(patient_name_pattern, text, re.IGNORECASE)
    if patient_name_match:
        result['patient_info']['name'] = patient_name_match.group(1).strip()
    
    patient_id_match = re.search(patient_id_pattern, text, re.IGNORECASE)
    if patient_id_match:
        result['patient_info']['id'] = patient_id_match.group(1).strip()
    
    patient_dob_match = re.search(patient_dob_pattern, text, re.IGNORECASE)
    if patient_dob_match:
        result['patient_info']['dob'] = patient_dob_match.group(1).strip()
    
    patient_age_match = re.search(patient_age_pattern, text, re.IGNORECASE)
    if patient_age_match:
        result['patient_info']['age'] = patient_age_match.group(1).strip()
    
    patient_gender_match = re.search(patient_gender_pattern, text, re.IGNORECASE)
    if patient_gender_match:
        gender = patient_gender_match.group(1).strip().upper()
        if gender in ['M', 'MALE']:
            result['patient_info']['gender'] = 'Male'
        elif gender in ['F', 'FEMALE']:
            result['patient_info']['gender'] = 'Female'
    
    # Extract dates
    dates = extract_dates(text)
    if dates:
        result['metadata'].update(dates)
    
    # Extract examination type
    exam_type = extract_examination_type(text)
    if exam_type:
        result['metadata']['examination_type'] = exam_type
    
    return result

def extract_birads_score(text: str) -> Optional[str]:
    """
    Extract BIRADS score from medical text
    
    Args:
        text: Medical report text
        
    Returns:
        BIRADS score as string, or None if not found
    """
    # Standard BIRADS score pattern
    birads_pattern = r'(?:BIRADS|BI-RADS)[:\s]*([0-6])'
    birads_matches = re.search(birads_pattern, text, re.IGNORECASE)
    
    if birads_matches:
        return birads_matches.group(1)
    
    # Alternative pattern for longer form
    alt_pattern = r'(?:BIRADS|BI-RADS)[:\s]*(Category\s*[0-6])'
    alt_matches = re.search(alt_pattern, text, re.IGNORECASE)
    
    if alt_matches:
        score = alt_matches.group(1)
        # Extract just the number
        number_match = re.search(r'(\d)', score)
        if number_match:
            return number_match.group(1)
        return score
    
    return None

def extract_dates(text: str) -> Dict[str, str]:
    """
    Extract relevant dates from medical text
    
    Args:
        text: Medical report text
        
    Returns:
        Dictionary with date fields
    """
    result = {}
    
    # Common date formats
    date_pattern = r'(?:\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}|\d{4}[-/\.]\d{1,2}[-/\.]\d{1,2})'
    
    # Document date patterns
    doc_date_pattern = r'(?:DATE|REPORT DATE|DATE OF REPORT)[:\s]+(' + date_pattern + ')'
    doc_date_match = re.search(doc_date_pattern, text, re.IGNORECASE)
    
    if doc_date_match:
        result['document_date'] = doc_date_match.group(1)
    
    # Examination date patterns
    exam_date_pattern = r'(?:EXAM DATE|DATE OF EXAM|EXAMINATION DATE|STUDY DATE)[:\s]+(' + date_pattern + ')'
    exam_date_match = re.search(exam_date_pattern, text, re.IGNORECASE)
    
    if exam_date_match:
        result['examination_date'] = exam_date_match.group(1)
    
    # If we haven't found explicit exam date, look for any date
    if 'examination_date' not in result:
        date_matches = re.findall(date_pattern, text)
        if date_matches:
            # Use the first date found as a fallback
            result['date_found'] = date_matches[0]
    
    return result

def extract_examination_type(text: str) -> Optional[str]:
    """
    Extract the type of medical examination from the text
    
    Args:
        text: Medical report text
        
    Returns:
        Examination type or None if not found
    """
    # Common examination types
    exam_types = [
        "MAMMOGRAM", "MAMMOGRAPHY", 
        "ULTRASOUND", "SONOGRAM", 
        "MRI", "MAGNETIC RESONANCE IMAGING",
        "CT", "CT SCAN", "COMPUTED TOMOGRAPHY",
        "X-RAY", "RADIOGRAPH",
        "PET SCAN", "POSITRON EMISSION TOMOGRAPHY",
        "BONE DENSITY", "DEXA", "DENSITOMETRY"
    ]
    
    # Create pattern to match any of the exam types
    pattern = r'(?:EXAMINATION|EXAM|PROCEDURE|STUDY)[:\s]+(?:(\w+\s*\w*\s*\w*\s*\w*))'
    exam_match = re.search(pattern, text, re.IGNORECASE)
    
    if exam_match:
        found_exam = exam_match.group(1).strip().upper()
        return found_exam
    
    # If no match with pattern, check if any exam type is mentioned in the text
    for exam_type in exam_types:
        if re.search(r'\b' + re.escape(exam_type) + r'\b', text, re.IGNORECASE):
            return exam_type
    
    return None

def create_structured_dataframe(extracted_data: Dict[str, Any], filename: str) -> List[Dict[str, Any]]:
    """
    Create a structured dataframe representation from extracted data
    
    Args:
        extracted_data: Dictionary of extracted data
        filename: Original filename
        
    Returns:
        List of dictionaries representing rows for a DataFrame
    """
    # Create a row for the dataframe
    row = {
        "Filename": filename,
        "Document Date": extracted_data.get('document_date', None),
        "Examination Date": extracted_data.get('examination_date', None),
        "Examination Type": extracted_data.get('examination_type', None),
        "BIRADS Score": extracted_data.get('birads_score', None),
        "Patient Name": None,
        "Patient ID": None,
        "Patient Age": None,
        "Patient Gender": None,
        "Findings": None,
        "Impression": None,
    }
    
    # Add patient info if available
    if 'patient_info' in extracted_data and extracted_data['patient_info']:
        patient_info = extracted_data['patient_info']
        row["Patient Name"] = patient_info.get('name', None)
        row["Patient ID"] = patient_info.get('id', None)
        row["Patient Age"] = patient_info.get('age', None)
        row["Patient Gender"] = patient_info.get('gender', None)
    
    # Add section content if available
    if 'sections' in extracted_data and extracted_data['sections']:
        sections = extracted_data['sections']
        row["Findings"] = sections.get('findings', None)
        row["Impression"] = sections.get('impression', None)
    
    # Get metadata if available
    if 'metadata' in extracted_data and extracted_data['metadata']:
        metadata = extracted_data['metadata']
        if 'document_date' in metadata and not row["Document Date"]:
            row["Document Date"] = metadata['document_date']
        if 'examination_date' in metadata and not row["Examination Date"]:
            row["Examination Date"] = metadata['examination_date']
        if 'examination_type' in metadata and not row["Examination Type"]:
            row["Examination Type"] = metadata['examination_type']
    
    # Override with direct fields if available
    if 'findings' in extracted_data and extracted_data['findings'] and not row["Findings"]:
        row["Findings"] = extracted_data['findings']
    if 'impression' in extracted_data and extracted_data['impression'] and not row["Impression"]:
        row["Impression"] = extracted_data['impression']
    if 'examination_type' in extracted_data and extracted_data['examination_type'] and not row["Examination Type"]:
        row["Examination Type"] = extracted_data['examination_type']
    
    return [row] 