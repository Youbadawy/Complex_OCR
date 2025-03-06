import cv2
import pytesseract
import re
import numpy as np
import logging
import asyncio
import json
from functools import lru_cache
from typing import Dict, Any, Optional, List, Tuple
from spellchecker import SpellChecker
from pathlib import Path
from pytesseract import Output

# Constants and paths
MEDICAL_TERMS_PATH = Path(__file__).parent / "medical_terms.txt"
FRENCH_MEDICAL_TERMS_PATH = Path(__file__).parent / "french_medical_terms.txt"

# Global caches
IMAGE_CACHE = {}
OCR_CACHE = {}  # Added OCR cache
LLM_CACHE = {}  # Added LLM cache
PRE_COMPILED_PATTERNS = {
    # Enhanced patterns with more format variations
    'impression': re.compile(r'\b(?:IMPRESSION|IMPRESSIONS|ASSESSMENT|CONCLUSION|INTERPRETATION|SUMMARY)[:.\s]*([^\n]+(?:\n[^\n]+)*)', re.IGNORECASE),
    'findings': re.compile(r'\b(?:FINDINGS|FINDING|OBSERVATION|REPORT|RESULTS|INTERPRETATION|RADIOLOGIC\s+FINDINGS)[:.\s]*([^\n]+(?:\n[^\n]+)*)', re.IGNORECASE),
    'recommendation': re.compile(r'\b(?:RECOMMENDATION|RECOMMENDATIONS|FOLLOW[\s-]*UP|ADVISED|SUGGEST(?:ED|ION)|PLAN)[:.\s]*([^\n]+(?:\n[^\n]+)*)', re.IGNORECASE),
    
    # Enhanced BIRADS pattern with subcategories and variations
    'birads': re.compile(r'(?i)(?:bi-rads|birads|category|assessment|classification)[\s:-]*(\d|[0-6][abc]?|zero|one|two|three|four|five|six|negative|benign|suspicious|malignant)', re.IGNORECASE),
    
    # Enhanced date patterns
    'date': re.compile(r'\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b|\b\d{1,2}[-/]\d{1,2}[-/]\d{4}\b|\b\d{1,2}[-/]\d{1,2}[-/]\d{2}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{2}\b|\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\b', re.IGNORECASE),
    
    # Enhanced medical terms
    'medical_terms': re.compile(r'\b(birads|bi-rads|impression|mammogram|ultrasound|density|fibroglandular|microcalcification|mass|nodule|cyst|asymmetry|lesion)\b', re.IGNORECASE),
    
    # Enhanced provider patterns
    'provider': re.compile(r'(?:electronically\s+signed\s+by|digitally\s+signed\s+by|dictated\s+by|physician|provider|radiologist|reported\s+by|interpreted\s+by|reading\s+radiologist)[:\s]+([A-Za-z\s.,]+(?:MD|M\.D\.|FRCPC|DO|PhD)?)', re.IGNORECASE),
    
    # Enhanced patient name pattern
    'patient_name': re.compile(r'(?:patient|name|patient name|patient information)[:\s]+([A-Za-z\s.-]+)(?:\s*\d|$)', re.IGNORECASE),
    
    # Enhanced exam date pattern
    'exam_date': re.compile(r'(?:exam date|date of exam|study date|document date|examination date)[:\s]+(\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{4}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4})', re.IGNORECASE),
    
    # Enhanced document type pattern
    'document_type': re.compile(r'\b(mammogram|mammography|ultrasound|breast imaging|screening|diagnostic|MAMMO|breast exam|breast study|MRI|magnetic resonance|breast ultrasound|sonogram|sonography)\b', re.IGNORECASE),
    
    # Enhanced deidentified information pattern
    'deidentified': re.compile(r'\b(PROTECTED\s+[A-Z]|REDACTED|\[REDACTED\]|CONFIDENTIAL|\[\*+\]|\*{3,}|\[(?:NAME|DOB|MRN|DATE|ID|PATIENT|ADDRESS)\]|DE-IDENTIFIED)\b', re.IGNORECASE),
    
    # Added patterns for signature blocks
    'signature_block': re.compile(r'(?:Electronically|Digitally)\s+(?:Signed|Approved|Verified)(?:\s+by)?[:\s]*.+?(?:MD|M\.D\.|PhD|FRCPC).+?(?:\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}|\d{4}[-/\.]\d{1,2}[-/\.]\d{1,2})', re.IGNORECASE | re.DOTALL),
    
    # Added patterns for disclaimers
    'disclaimer': re.compile(r'(?:DISCLAIMER|LEGAL\s+NOTICE|PRIVACY\s+NOTICE|Confidentiality\s+Notice|This\s+(?:document|report|information)\s+(?:may\s+contain|is)\s+(?:confidential|private|privileged)|CONFIDENTIAL.*?NOT\s+FOR\s+DISTRIBUTION|Density\s+was\s+assessed\s+by.*?(?:software|algorithm|system))', re.IGNORECASE | re.DOTALL),
    
    # Added patterns for patient history
    'patient_history': re.compile(r'\b(?:HISTORY|CLINICAL\s+HISTORY|INDICATION|CLINICAL\s+INDICATION|REASON\s+FOR\s+(?:EXAM|STUDY|IMAGING)|PRESENTING\s+COMPLAINT)[:.\s]*([^\n]+(?:\n[^\n]+)*)', re.IGNORECASE),
}

def _cache_key(image: np.ndarray) -> str:
    """Generate cache key from image data"""
    from hashlib import sha256
    return sha256(image.tobytes()).hexdigest()

def check_tesseract_installation():
    """Check if Tesseract is properly installed"""
    try:
        # Try to get tesseract version
        version = pytesseract.get_tesseract_version()
        logging.info(f"Tesseract version: {version}")
        return True
    except Exception as e:
        logging.error(f"Tesseract check failed: {str(e)}")
        return False
def simple_ocr(image: np.ndarray) -> str:
    """Reliable OCR pipeline with basic preprocessing"""
    try:
        # Check if Tesseract is installed
        if not check_tesseract_installation():
            return "[OCR ERROR: Tesseract not installed. Please install Tesseract from https://github.com/UB-Mannheim/tesseract/wiki]"
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Basic adaptive thresholding
        processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 11, 2)
        
        # OCR with Tesseract
        text_data = pytesseract.image_to_data(
            processed,
            output_type=Output.DICT,
            config='--psm 3 --oem 3'  # Fully automatic page segmentation
        )
        
        # Filter by confidence
        text = ' '.join([
            text_data['text'][i] 
            for i in range(len(text_data['text'])) 
            if int(text_data['conf'][i]) > 60
        ])
        
        return text.strip()
    
    except Exception as e:
        logging.error(f"OCR failed: {str(e)}")
        return f"[OCR ERROR: {str(e)}]"

def normalize_date(date_str: str) -> str:
    """
    Normalize date string to YYYY-MM-DD format with French support.
    
    Args:
        date_str: Date string in various formats
        
    Returns:
        Normalized date string in YYYY-MM-DD format or original string if parsing fails
    """
    if not date_str or not isinstance(date_str, str):
        return ""
    
    # Add French date patterns
    date_patterns = [
        # YYYY-MM-DD or YYYY/MM/DD
        (r'(\d{4})[-/](\d{1,2})[-/](\d{1,2})', 
         lambda m: f"{m.group(1)}-{m.group(2).zfill(2)}-{m.group(3).zfill(2)}"),
        
        # DD/MM/YYYY (French format) or MM/DD/YYYY (US format)
        (r'(\d{1,2})[-/](\d{1,2})[-/](\d{4})', 
         lambda m: _parse_date_with_locale(m.group(1), m.group(2), m.group(3))),
        
        # French date format: DD month YYYY
        (r'(\d{1,2})\s+(janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)\s+(\d{4})',
         lambda m: _parse_french_text_date(m.group(1), m.group(2), m.group(3))),
        
        # French date with month abbreviations
        (r'(\d{1,2})\s+(janv|févr|mars|avr|mai|juin|juil|août|sept|oct|nov|déc)\.?\s+(\d{4})',
         lambda m: _parse_french_text_date(m.group(1), m.group(2), m.group(3))),
        
        # French date with "le" prefix
        (r'le\s+(\d{1,2})\s+(janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)\s+(\d{4})',
         lambda m: _parse_french_text_date(m.group(1), m.group(2), m.group(3))),
        
        # English date format: Month DD, YYYY
        (r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2})(?:,|\s+)?\s*(\d{4})',
         lambda m: _parse_english_text_date(m.group(2), m.group(1), m.group(3))),
        
        # With month abbreviations: MMM DD, YYYY
        (r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?\s+(\d{1,2})(?:,|\s+)?\s*(\d{4})',
         lambda m: _parse_english_text_date(m.group(2), m.group(1), m.group(3))),
    ]
    
    # Try each pattern
    for pattern, formatter in date_patterns:
        match = re.search(pattern, date_str, re.IGNORECASE)
        if match:
            try:
                return formatter(match)
            except Exception as e:
                logging.warning(f"Date parsing error: {str(e)}")
                continue
    
    # If standard patterns fail, try dateutil parser with locale
    try:
        from dateutil import parser
        # Try to detect if it's a French date for locale setting
        has_french = bool(re.search(r'janv|févr|mars|avr|juin|juil|août|sept|oct|nov|déc|le \d{1,2}', 
                                    date_str, re.IGNORECASE))
        
        if has_french:
            import locale
            old_locale = locale.setlocale(locale.LC_TIME)
            try:
                locale.setlocale(locale.LC_TIME, 'fr_FR.UTF-8')  # Set French locale
                parsed_date = parser.parse(date_str, fuzzy=True)
                return parsed_date.strftime('%Y-%m-%d')
            except Exception:
                # If fr_FR.UTF-8 fails, try fr_FR
                try:
                    locale.setlocale(locale.LC_TIME, 'fr_FR')
                    parsed_date = parser.parse(date_str, fuzzy=True)
                    return parsed_date.strftime('%Y-%m-%d')
                except Exception:
                    # If still fails, try without locale
                    parsed_date = parser.parse(date_str, fuzzy=True)
                    return parsed_date.strftime('%Y-%m-%d')
            finally:
                try:
                    locale.setlocale(locale.LC_TIME, old_locale)  # Restore original locale
                except Exception:
                    pass  # Ignore if we can't restore locale
        else:
            parsed_date = parser.parse(date_str, fuzzy=True)
            return parsed_date.strftime('%Y-%m-%d')
    except Exception as e:
        logging.warning(f"Date parsing fallback error: {str(e)}")
        return date_str

def _parse_french_text_date(day, month, year):
    """
    Parse a French text date into YYYY-MM-DD format
    
    Args:
        day: Day as string
        month: Month name in French
        year: Year as string
        
    Returns:
        Date in YYYY-MM-DD format
    """
    # Map French month names to numbers
    month_map = {
        'janvier': '01', 'janv': '01',
        'février': '02', 'févr': '02',
        'mars': '03',
        'avril': '04', 'avr': '04',
        'mai': '05',
        'juin': '06',
        'juillet': '07', 'juil': '07',
        'août': '08',
        'septembre': '09', 'sept': '09',
        'octobre': '10', 'oct': '10',
        'novembre': '11', 'nov': '11',
        'décembre': '12', 'déc': '12'
    }
    
    month_num = month_map.get(month.lower(), '01')  # Default to 01 if not found
    return f"{year}-{month_num}-{day.zfill(2)}"

def _parse_english_text_date(day, month, year):
    """
    Parse an English text date into YYYY-MM-DD format
    
    Args:
        day: Day as string
        month: Month name in English
        year: Year as string
        
    Returns:
        Date in YYYY-MM-DD format
    """
    # Map English month names to numbers
    month_map = {
        'january': '01', 'jan': '01',
        'february': '02', 'feb': '02',
        'march': '03', 'mar': '03',
        'april': '04', 'apr': '04',
        'may': '05',
        'june': '06', 'jun': '06',
        'july': '07', 'jul': '07',
        'august': '08', 'aug': '08',
        'september': '09', 'sep': '09',
        'october': '10', 'oct': '10',
        'november': '11', 'nov': '11',
        'december': '12', 'dec': '12'
    }
    
    month_num = month_map.get(month.lower(), '01')  # Default to 01 if not found
    return f"{year}-{month_num}-{day.zfill(2)}"

def _parse_date_with_locale(num1, num2, year):
    """
    Parse a date with ambiguous MM/DD or DD/MM format based on value logic.
    Assumes DD/MM/YYYY format (French) when both numbers could be valid.
    
    Args:
        num1: First number in the date string
        num2: Second number in the date string
        year: Year as string
        
    Returns:
        Date in YYYY-MM-DD format
    """
    n1, n2 = int(num1), int(num2)
    
    # If first number is > 12, it must be a day
    if n1 > 12:
        return f"{year}-{str(n2).zfill(2)}-{str(n1).zfill(2)}"
    
    # If second number is > 12, it must be a day
    if n2 > 12:
        return f"{year}-{str(n1).zfill(2)}-{str(n2).zfill(2)}"
    
    # If both could be months, assume DD/MM format (French)
    return f"{year}-{str(n2).zfill(2)}-{str(n1).zfill(2)}"

def extract_medical_fields(text, use_llm=False, llm_api_key=None):
    """
    Extract medical fields from document text with enhanced extraction and validation
    
    Args:
        text: Document text
        use_llm: Whether to use LLM for enhancing extraction
        llm_api_key: Optional API key for LLM service
        
    Returns:
        Dictionary with extracted fields
    """
    try:
        from document_processing.text_analysis import (
            extract_patient_info, extract_exam_type, extract_date_from_text,
            extract_provider_info, extract_signature_block, extract_birads_score,
            extract_sections, is_redacted_document, validate_field_types,
            enhance_extraction_with_llm, get_extraction_value, extract_section,
            extract_exam_date  # Add the new function
        )
        
        # Initialize result dictionary with empty fields
        result = {
            'patient_name': '',
            'mrn': '',
            'dob': '',
            'age': '',
            'gender': '',
            'exam_date': '',
            'exam_type': '',
            'document_type': '',
            'birads_score': '',
            'impression_result': '',
            'mammogram_results': '',
            'ultrasound_results': '',
            'recommendation': '',
            'patient_history': '',
            'electronically_signed_by': '',
            'testing_provider': '',
            'document_date': '',
            'referring_provider': '',
            'facility_name': '',  # Added facility name field
            'is_redacted': False
        }
        
        # First check if document is redacted
        try:
            is_redacted = is_redacted_document(text)
            result['is_redacted'] = is_redacted
        except Exception as e:
            logging.error(f"Error checking if document is redacted: {str(e)}")
            result['is_redacted'] = False
        
        # Extract structured data
        structured_data = {}
        
        # Patient information with structured extraction
        try:
            patient_info = extract_patient_info(text)
            structured_data.update(patient_info)
        except Exception as e:
            logging.error(f"Error extracting patient info: {str(e)}")
        
        # Exam type and date with structured extraction
        try:
            structured_data['exam_type'] = extract_exam_type(text)
        except Exception as e:
            logging.error(f"Error extracting exam type: {str(e)}")
        
        # Use the enhanced exam date extraction
        try:
            exam_date_data = extract_exam_date(text)
            if exam_date_data and exam_date_data.get('value'):
                structured_data['exam_date'] = exam_date_data
            else:
                # Fallback to traditional date extraction
                if 'exam_date' not in structured_data or not structured_data.get('exam_date', {}).get('value'):
                    date_extraction = extract_date_from_text(text)
                    if date_extraction and 'value' in date_extraction:
                        structured_data['exam_date'] = date_extraction
        except Exception as e:
            logging.error(f"Error extracting exam date: {str(e)}")
            
            # If both methods fail, use a simple date extraction as last resort
            try:
                if 'exam_date' not in structured_data or not structured_data.get('exam_date', {}).get('value'):
                    # Look for dates near exam keywords
                    date_matches = re.findall(r'(?:exam|examination|date).*?(\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{4})', 
                                            text, re.IGNORECASE)
                    if date_matches:
                        normalized_date = normalize_date(date_matches[0])
                        if normalized_date:
                            structured_data['exam_date'] = {"value": normalized_date, "confidence": 0.5}
            except Exception as nested_e:
                logging.error(f"Error in date fallback extraction: {str(nested_e)}")
        
        # Extract provider information with enhanced function
        try:
            provider_info = extract_provider_info(text)
            structured_data.update(provider_info)
        except Exception as e:
            logging.error(f"Error extracting provider info: {str(e)}")
        
        # Extract signature block
        try:
            signature_block = extract_signature_block(text)
            if signature_block and 'value' in signature_block:
                structured_data['signature_block'] = signature_block
        except Exception as e:
            logging.error(f"Error extracting signature block: {str(e)}")
        
        # Extract BIRADS score
        try:
            birads_score = extract_birads_score(text)
            if birads_score and 'value' in birads_score:
                structured_data['birads_score'] = birads_score
        except Exception as e:
            logging.error(f"Error extracting BIRADS score: {str(e)}")
            
        # Extract sections using the new extract_section function
        try:
            structured_data['findings'] = {'value': extract_section(text, 'findings'), 'confidence': 0.8}
            structured_data['impression_result'] = {'value': extract_section(text, 'impression'), 'confidence': 0.8}
            structured_data['recommendation'] = {'value': extract_section(text, 'recommendation'), 'confidence': 0.8}
            
            # Extract and merge clinical_history and patient_history
            clinical_history = extract_section(text, 'clinical_history')
            patient_history = extract_section(text, 'patient_history')
            
            if clinical_history and patient_history:
                # Merge if both exist
                structured_data['patient_history'] = {
                    'value': f"{clinical_history}\n{patient_history}",
                    'confidence': 0.8
                }
            elif clinical_history:
                structured_data['patient_history'] = {'value': clinical_history, 'confidence': 0.8}
            elif patient_history:
                structured_data['patient_history'] = {'value': patient_history, 'confidence': 0.8}
        except Exception as e:
            logging.error(f"Error extracting sections: {str(e)}")
        
        # Apply field validations
        try:
            structured_data = validate_field_types(structured_data)
        except Exception as e:
            logging.error(f"Error validating field types: {str(e)}")
        
        # Use LLM to enhance extraction if specified or if critical fields are missing/low confidence
        should_use_llm = use_llm

        # Auto-trigger LLM if critical fields are missing or have very low confidence
        try:
            critical_fields = ['patient_name', 'exam_date', 'document_date', 'exam_type']
            missing_critical = any(field not in structured_data for field in critical_fields)
            low_confidence_critical = any(
                field in structured_data and 
                isinstance(structured_data[field], dict) and 
                structured_data[field].get('confidence', 1.0) < 0.4
                for field in critical_fields
            )
            
            if missing_critical or low_confidence_critical:
                should_use_llm = True
                
            if should_use_llm:
                structured_data = enhance_extraction_with_llm(structured_data, text, llm_api_key)
        except Exception as e:
            logging.error(f"Error enhancing extraction with LLM: {str(e)}")
            
        # Always perform a final validation with LLM to check if results make sense
        logging.info("Validating extraction results with LLM")
        try:
            from document_processing.text_analysis import extract_with_llm
            # Get LLM view of the document
            llm_view = extract_with_llm(text, llm_api_key)
            
            # Check for any major discrepancies
            for field in critical_fields:
                if field in structured_data and field in llm_view:
                    structured_value = structured_data[field].get('value', '') if isinstance(structured_data[field], dict) else ''
                    llm_value = llm_view[field].get('value', '') if isinstance(llm_view[field], dict) else ''
                    
                    # If values are very different, log warning
                    if structured_value and llm_value and structured_value != llm_value:
                        logging.warning(f"Discrepancy in {field}: Regex extracted '{structured_value}' but LLM suggests '{llm_value}'")
                        
                        # If LLM has high confidence and regex has low confidence, prefer LLM
                        regex_confidence = structured_data[field].get('confidence', 0) if isinstance(structured_data[field], dict) else 0
                        llm_confidence = llm_view[field].get('confidence', 0) if isinstance(llm_view[field], dict) else 0
                        
                        if regex_confidence < 0.6 and llm_confidence > 0.7:
                            logging.info(f"Using LLM value for {field} due to higher confidence")
                            structured_data[field] = llm_view[field]
        except Exception as e:
            logging.error(f"Error during final LLM validation: {str(e)}")
        
        # Map structured data to simple format for UI
        try:
            result['patient_name'] = get_extraction_value(structured_data, 'patient_name')
            result['mrn'] = get_extraction_value(structured_data, 'mrn')
            result['dob'] = get_extraction_value(structured_data, 'dob')
            result['age'] = get_extraction_value(structured_data, 'age')
            result['gender'] = get_extraction_value(structured_data, 'gender')
            
            # Normalize dates
            exam_date = get_extraction_value(structured_data, 'exam_date')
            result['exam_date'] = normalize_date(exam_date) if exam_date else ''
            
            document_date = get_extraction_value(structured_data, 'document_date')
            result['document_date'] = normalize_date(document_date) if document_date else ''
            
            result['exam_type'] = get_extraction_value(structured_data, 'exam_type')
            result['document_type'] = get_extraction_value(structured_data, 'document_type')
            result['birads_score'] = get_extraction_value(structured_data, 'birads_score')
            result['impression_result'] = get_extraction_value(structured_data, 'impression_result')
            result['mammogram_results'] = get_extraction_value(structured_data, 'mammogram_results', text)
            result['ultrasound_results'] = get_extraction_value(structured_data, 'ultrasound_results')
            result['recommendation'] = get_extraction_value(structured_data, 'recommendation')
            result['patient_history'] = get_extraction_value(structured_data, 'patient_history')
            result['electronically_signed_by'] = get_extraction_value(structured_data, 'electronically_signed_by')
            result['testing_provider'] = get_extraction_value(structured_data, 'testing_provider')
            result['referring_provider'] = get_extraction_value(structured_data, 'referring_provider')
            result['facility_name'] = get_extraction_value(structured_data, 'facility_name')
        except Exception as e:
            logging.error(f"Error mapping structured data to UI format: {str(e)}")
        
        return result
        
    except Exception as e:
        logging.error(f"Error in extract_medical_fields: {str(e)}")
        # Return a basic result with redacted flag set
        return {
            'patient_name': 'REDACTED',
            'is_redacted': True,
            'exam_date': '',
            'exam_type': '',
            'document_type': '',
            'birads_score': '',
            'facility_name': ''
        }

# Preload medical dictionary for spell checking
MEDICAL_DICT = SpellChecker(language=None)
MEDICAL_DICT.word_frequency.load_text_file(str(MEDICAL_TERMS_PATH), encoding='utf-8')
if FRENCH_MEDICAL_TERMS_PATH.exists():
    MEDICAL_DICT.word_frequency.load_text_file(
        str(FRENCH_MEDICAL_TERMS_PATH), 
        encoding='latin-1'  # Changed from utf-8 to handle accented characters
    )
else:
    logging.warning("French medical terms file not found, using English only")

import aiohttp
from tenacity import AsyncRetrying
import json
import pandas as pd
from pytesseract import Output
import os
import numpy as np
import dotenv
import json
import requests
from typing import Dict, Any
from groq import Groq
from langdetect import detect, LangDetectException
from transformers import MarianMTModel, MarianTokenizer, pipeline as tf_pipeline
import torch
from spellchecker import SpellChecker
import logging
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
from tenacity import retry, stop_after_attempt, wait_fixed

# Constants for medical term validation
MEDICAL_TERMS = {"birads", "impression", "mammogram", "ultrasound"}

def hybrid_ocr(image: np.ndarray) -> str:
    """Hybrid OCR pipeline with medical validation"""
    # Enhanced preprocessing for medical documents
    enhanced = cv2.convertScaleAbs(image, alpha=1.5, beta=40)
    processed = preprocess_image(enhanced, apply_sharpen=True)
    
    # Attempt PaddleOCR first with validation
    paddle_text = ""
    try:
        if not hasattr(extract_text_paddle, "ocr_instance"):
            raise RuntimeError("PaddleOCR not initialized")
            
        # Health check with test image
        test_img = np.zeros((100,300,3), dtype=np.uint8)
        cv2.putText(test_img, "BIRADS", (10,50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        test_result = extract_text_paddle.ocr_instance.ocr(test_img)
        if not test_result or "BIRADS" not in str(test_result):
            raise RuntimeError("PaddleOCR health check failed")
            
        # Process actual image
        result = extract_text_paddle.ocr_instance.ocr(processed)
        paddle_text = ' '.join([line[1][0] for line in result[0]]) if result else ""
        
    except Exception as e:
        logging.warning(f"PaddleOCR failed: {str(e)} - Switching to Tesseract")
        paddle_text = ""

    # Fallback to Tesseract with enhanced processing
    try:
        tesseract_result = extract_text_tesseract(processed)
        tesseract_text = ' '.join([t for t, c in zip(tesseract_result['text'], 
                                  tesseract_result['confidence']) if c > 40])
    except Exception as e:
        logging.error(f"Tesseract failed: {str(e)}")
        tesseract_text = ""

    # Combine and validate results
    combined_text = f"{paddle_text} {tesseract_text}".strip()
    
    # Medical content validation with lower threshold
    if medical_term_score(combined_text) < 1:  # Require at least 1 medical term
        logging.warning("Low medical content - using raw OCR fallback")
        return f"{paddle_text} {tesseract_text}".strip()
    
    return combined_text

def validate_results(texts: dict) -> str:
    """Validate OCR results against medical terms"""
    corrections = {
        r"\bIMPRessiON\b": "IMPRESSION",
        r"\baimost\b": "almost",
        r"\bMamm0gram\b": "Mammogram"
    }
    
    # Ensure all values are strings
    validated_texts = {k: str(v) for k, v in texts.items()}
    
    best_text = max(validated_texts.values(), key=lambda t: medical_term_score(t))
    
    # Apply regex corrections with null checks
    if not isinstance(best_text, str):
        best_text = ""
        
    for pattern, replacement in corrections.items():
        try:
            best_text = re.sub(pattern, replacement, best_text, flags=re.IGNORECASE)
        except TypeError:
            best_text = ""
            logging.error("Invalid text type for regex substitution")
        
    return best_text

def medical_term_score(text: str) -> int:
    """Score text based on medical term presence"""
    if not isinstance(text, str):
        return 0
        
    return sum(
        1 for term in MEDICAL_TERMS
        if re.search(rf"\b{term}\b", text, re.IGNORECASE)
    )

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def extract_text_paddle(image: np.ndarray) -> str:
    """Extract text using PaddleOCR with health checks and retries"""
    try:
        # Use current initialization pattern
        if not hasattr(extract_text_paddle, "ocr_instance"):
            from paddleocr import PaddleOCR
            extract_text_paddle.ocr_instance = PaddleOCR(
                use_angle_cls=True,
                lang='en',
                use_gpu=False,
                enable_mkldnn=True
            )
            
        # Health check
        dummy_check = np.zeros((100,100,3), dtype=np.uint8)
        extract_text_paddle.ocr_instance.ocr(dummy_check, cls=True)
        
        # Process with validation
        result = extract_text_paddle.ocr_instance.ocr(image, cls=True)
        text = ' '.join([line[1][0] for line in result[0]]) if result else ""
        
        if not text.strip():
            raise ValueError("Empty OCR result")
            
        return text
        
    except Exception as e:
        logging.error(f"PaddleOCR critical failure: {str(e)}")
        raise

def preprocess_image(image, apply_sharpen=True, downscale_factor=1.0):
    """Optimized image preprocessing with caching"""
    cache_key = _cache_key(image) + f"_{apply_sharpen}_{downscale_factor}"
    
    if cache_key in IMAGE_CACHE: 
        return IMAGE_CACHE[cache_key]
    # Downscale if specified
    if downscale_factor < 1.0:
        h, w = image.shape[:2]
        image = cv2.resize(image, (int(w*downscale_factor), int(h*downscale_factor)))
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    # Fast contrast check
    contrast = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Conditional noise reduction
    if contrast < 50:
        if contrast < 30:  # Heavy noise
            gray = cv2.medianBlur(gray, 3)
        else:  # Moderate noise
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Deskewing optimization
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
    
    if lines is not None:
        angles = [np.degrees(np.arctan2(y2-y1, x2-x1)) for line in lines for x1,y1,x2,y2 in line]
        median_angle = np.median(angles)
        if abs(median_angle) > 1.0:  # Only deskew if angle > 1 degree
            h, w = gray.shape
            M = cv2.getRotationMatrix2D((w//2, h//2), median_angle, 1)
            gray = cv2.warpAffine(gray, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    # Adaptive thresholding
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY, 11, 2)
    
    # Conditional sharpening
    if apply_sharpen and contrast < 40:
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        binary = cv2.filter2D(binary, -1, kernel)
        
    IMAGE_CACHE[cache_key] = binary
    return binary

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def extract_text_tesseract(image):
    """Run Tesseract with multiple page segmentation modes and select best result"""
    psms = ['3', '6', '11']  # Auto, Single Block, Sparse Text
    all_results = []
    
    for psm in psms:
        try:
            data = pytesseract.image_to_data(
                image,
                output_type=Output.DICT,
                lang='fra+eng',
                timeout=30,  # Set 30 second timeout per PSM
                config=(
                    f'--psm {psm} '
                    '-c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-:.()%éèàçêâôûùîÉÈÀÇÊÂÔÛÙÎ" '
                    '--oem 3'
                )
            )
        except pytesseract.TesseractError as e:
            logging.warning(f"Tesseract PSM {psm} failed: {str(e)}")
            all_results.append({
                'text': [],
                'confidence': [],
                'psm': psm,
                'avg_conf': 0
            })
            continue
            
        # Calculate average confidence for this PSM
        valid_confs = [c for c in data['conf'] if c != -1]
        avg_conf = sum(valid_confs)/len(valid_confs) if valid_confs else 0
        all_results.append({
            'text': data['text'],
            'confidence': data['conf'],
            'psm': psm,
            'avg_conf': avg_conf
        })
    
    # Add validation for empty results
    if not all_results or all(r['avg_conf'] == 0 for r in all_results):
        logging.error("All Tesseract PSMs failed")
        return {'text': [], 'confidence': [], 'psm': '0'}
    
    # Select best result with confidence check
    best_result = max(all_results, key=lambda x: x['avg_conf'])
    if best_result['avg_conf'] < 20:  # Minimum confidence threshold
        raise OCRError("Low confidence OCR result")
    
    return {
        'text': best_result['text'],
        'confidence': best_result['confidence'],
        'psm': best_result['psm']
    }

def select_best_ocr_result(results):
    """Select best OCR result using hybrid scoring (70% LLM coherence, 30% confidence)"""
    text_options = [f"Option {i+1} (PSM {r['psm']}): {' '.join(r['text'])}" 
                   for i, r in enumerate(results)]
    
    try:
        # Get LLM scores for each option
        llm_response = parse_text_with_llm(
            f"Score each OCR option (1-10) for coherence and medical field presence. "
            f"Return JSON with 'scores': [int, ...].\n\n" + '\n\n'.join(text_options)
        )
        llm_scores = llm_response.get('scores', [5] * len(results))  # Default to mid-score if missing
        
        # Calculate hybrid scores (LLM 70% + Confidence 30%)
        hybrid_scores = [
            (0.7 * (llm / 10) + 0.3 * (r['avg_conf'] / 100), i)  # Normalize both to 0-1 scale
            for i, (llm, r) in enumerate(zip(llm_scores, results))
        ]
        
        # Get best index from hybrid scores
        best_idx = max(hybrid_scores, key=lambda x: x[0])[1]
        best_result = results[best_idx]
        
        # Verify minimum field presence
        text = ' '.join(best_result['text']).lower()
        required_terms = r'(patient|date|birads)'
        if not re.search(required_terms, text):
            logging.warning("Best OCR result lacks required fields, using confidence fallback")
            return max(results, key=lambda x: x['avg_conf'])
            
        return best_result
        
    except Exception as e:
        logging.warning(f"Hybrid selection failed: {str(e)}")
        return max(results, key=lambda x: x['avg_conf'])

def parse_extracted_text(ocr_result):
    raw_text = ' '.join([t for t, c in zip(ocr_result['text'], ocr_result['confidence']) if c != -1])
    
    # Apply critical medical corrections first
    raw_text = PRE_COMPILED_PATTERNS['impression'].sub(r'IMPRESSION: \1', raw_text)
    raw_text = PRE_COMPILED_PATTERNS['birads'].sub(r'BIRADS \2', raw_text)
    
    # Fast spell checking only for non-medical terms
    words = raw_text.split()
    corrected = []
    
    for word in words:
        # Skip numbers and already valid medical terms
        if word.isdigit() or PRE_COMPILED_PATTERNS['medical_terms'].match(word):
            corrected.append(word)
            continue
            
        # Check spelling only if not in medical dictionary
        if not MEDICAL_DICT.known([word.lower()]):
            candidates = MEDICAL_DICT.candidates(word)
            if candidates:
                word = max(candidates, key=MEDICAL_DICT.word_usage_frequency)
                
        corrected.append(word)
    
    # Apply remaining regex patterns
    structured_text = '\n'.join(corrected)
    structured_text = PRE_COMPILED_PATTERNS['date'].sub(
        lambda m: standardize_date(m.group()), 
        structured_text
    )
    
    # Clean and structure text from complex layouts
    lines = []
    current_line = []
    for i, word in enumerate(raw_text.split()):
        # Correct spelling using medical dictionary
        if not MEDICAL_DICT.known([word.lower()]):
            corrected = MEDICAL_DICT.correction(word)
            if corrected and corrected != word:
                word = corrected
        
        if word.isupper() and len(current_line) > 0:
            lines.append(' '.join(current_line))
            current_line = [word]
        else:
            current_line.append(word)
    lines.append(' '.join(current_line))
    
    cleaned_text = '\n'.join(filter(None, [
        line.strip(' -•*®©™§¶†‡')
        for line in lines
        if line.strip() and len(line.strip()) > 2
    ]))
    
    # Standardize medical terms
    cleaned_text = re.sub(r'(?i)bi-rads', 'BIRADS', cleaned_text)
    cleaned_text = re.sub(r'(?i)mammo\s*-?\s*', 'MAMMO - ', cleaned_text)
    
    # Improved date validation
    date_matches = re.finditer(
        r'\b(20\d{2}(?:-(0[1-9]|1[0-2])(?:-(0[1-9]|[12][0-9]|3[01]))?)?\b'  # YYYY-MM-DD
        r'|(0?[1-9]|1[0-2])[/-](0[1-9]|[12][0-9]|3[01])[/-](20\d{2})'  # MM/DD/YYYY
        r'|(0[1-9]|[12][0-9]|3[01])[-/](0[1-9]|1[0-2])[-/](20\d{2})',  # DD-MM-YYYY
        cleaned_text
    )
    
    for match in date_matches:
        try:
            std_date = pd.to_datetime(match.group()).strftime('%Y-%m-%d')
            cleaned_text = cleaned_text.replace(match.group(), std_date)
        except pd.errors.OutOfBoundsDatetime:
            logging.warning(f"Invalid date format: {match.group()}")
            continue
    
    return cleaned_text

def load_templates():
    """Load templates with path validation"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    template_dir = os.path.join(current_dir, "templates")
    
    if not os.path.exists(template_dir):
        logging.error(f"Template directory not found: {template_dir}")
        return {}
    
    templates = {}
    for filename in os.listdir(template_dir):
        if filename.endswith(".png"):
            path = os.path.join(template_dir, filename)
            try:
                template = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if template is None:
                    raise ValueError("Invalid template image")
                key = filename.replace("_template.png", "")
                templates[key] = template
            except Exception as e:
                logging.warning(f"Failed to load {filename}: {str(e)}")
                
    return templates

TEMPLATES = load_templates()

class OCRError(Exception):
    """Custom exception for OCR processing errors"""
    pass

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from groq import Groq

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=5, min=5, max=60),
    retry=retry_if_exception_type((
        requests.exceptions.HTTPError,
        json.JSONDecodeError,
        KeyError,
        Exception
    )),
    before_sleep=lambda _: logging.warning("Rate limited, retrying...")
)
async def parse_text_with_llm_async(text: str) -> Dict[str, Any]:
    """Async version of LLM parsing with rate limiting"""
    semaphore = asyncio.Semaphore(5)  # Max 5 concurrent requests
    async with semaphore:
        async with aiohttp.ClientSession() as session:
            async for attempt in AsyncRetrying(stop=stop_after_attempt(3)):
                with attempt:
                    prompt = create_llm_prompt(text)
                    async with session.post(
                        "https://api.groq.com/v1/chat/completions",
                        headers={"Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}"},
                        json={
                            "messages": [{"role": "user", "content": prompt}],
                            "model": "mixtral-8x7b-32768",
                            "temperature": 0.2
                        }
                    ) as response:
                        response.raise_for_status()
                        return await handle_llm_response(await response.json())

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=5, min=5, max=60),
    retry=retry_if_exception_type((
        requests.exceptions.HTTPError,
        json.JSONDecodeError,
        KeyError,
        Exception
    )),
    before_sleep=lambda _: logging.warning("Rate limited, retrying...")
)
def parse_text_with_llm(text: str) -> Dict[str, Any]:
    """Extract structured fields from OCR text using Groq's LLM API"""
    dotenv.load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        logging.error("Groq API key not found in .env file")
        raise OCRError("API configuration error")

    client = Groq(api_key=api_key)
    
    prompt = f"""Extract structured medical information from this mammogram report text.
Return JSON with these fields:
- patient_name
- exam_date (YYYY-MM-DD format)
- document_date (YYYY-MM-DD format)
- document_type (e.g., Mammogram, Ultrasound)
- birads_score (0-6)
- impression_result
- mammogram_results
- ultrasound_results
- patient_history
- recommendation
- electronically_signed_by
- testing_provider
- confidence (0-100)

Report text:
{text[:3000]}"""

    try:
        # Log the full prompt and input text
        logging.info(f"LLM Prompt:\n{prompt}")
        logging.debug(f"Full Input Text:\n{text[:3000]}")  # Truncated to 3000 chars
        
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="mixtral-8x7b-32768",
            temperature=0.2,
            max_tokens=2000,
            response_format={"type": "json_object"}
        )
        
        # Validate Groq API response structure
        if not response.choices or not hasattr(response.choices[0].message, 'content'):
            raise ValueError("Invalid Groq API response format")
        
        # Log the raw API response
        raw_response = response.choices[0].message.content
        logging.info(f"Raw API Response:\n{raw_response}")
        
        result = json.loads(raw_response)
        
        # Validate response structure
        required = ['patient_name', 'exam_date', 'birads_score',
                   'impression_result', 'recommendation', 'confidence']
        if all(k in result for k in required) and isinstance(result['confidence'], (int, float)):
            return result
            
        raise ValueError("Invalid response structure from LLM")
            
    except Exception as e:
        logging.error(f"LLM extraction failed: {str(e)}")
        raise OCRError("Failed to extract fields after 3 attempts") from e

def create_llm_prompt(text: str) -> str:
    """Create a standardized prompt for LLM processing"""
    return f"""Extract structured medical information from this mammogram report text.
Focus on patient details, exam dates, BIRADS scores, and key findings.

Report text:
{text}

Return a JSON response with these fields:
- patient_name
- exam_date 
- document_date
- document_type
- birads_score
- impression_result
- mammogram_results
- ultrasound_results
- patient_history
- recommendation
- electronically_signed_by
- testing_provider
"""

async def handle_llm_response(response_json: dict) -> Dict[str, Any]:
    """Process and validate LLM API response"""
    if not isinstance(response_json, dict):
        raise ValueError("Invalid response format")
    
    # Extract choices array
    choices = response_json.get('choices', [])
    if not choices:
        raise ValueError("Empty response from LLM")
        
    # Get content from first choice
    content = choices[0].get('message', {}).get('content', '')
    if not content:
        raise ValueError("No content in LLM response")
        
    # Parse JSON content
    try:
        result = json.loads(content)
        return result
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON in LLM response")

def validate_llm_results(data: Dict[str, Any]) -> bool:
    """Validate LLM extraction results"""
    required_fields = [
        'patient_name', 'exam_date', 'birads_score',
        'impression_result', 'recommendation'
    ]
    
    # Check all required fields exist and aren't None
    if not all(field in data for field in required_fields):
        return False
        
    # Validate BIRADS score is valid integer 0-6
    try:
        score = int(data['birads_score'])
        if not 0 <= score <= 6:
            return False
    except (ValueError, TypeError):
        return False
            
    return True

def extract_fields_from_text(text: str) -> Dict[str, Any]:
    """Extract structured fields with improved error handling"""
    # First try regex extraction for all fields
    regex_fields = extract_medical_fields(text)
    
    # Check if we have all required fields
    required_fields = ['patient_name', 'exam_date', 'birads_score']
    missing_fields = [f for f in required_fields if not regex_fields.get(f)]
    
    # If missing critical fields, try LLM extraction
    if missing_fields:
        try:
            llm_data = parse_text_with_llm(text)
            if validate_llm_results(llm_data):
                # Update missing fields from LLM results
                for field in missing_fields:
                    if field in llm_data and llm_data[field]:
                        regex_fields[field] = llm_data[field]
                
                # Also update these fields which LLM might extract better
                for field in ['impression_result', 'recommendation', 'patient_history']:
                    if field in llm_data and llm_data[field]:
                        regex_fields[field] = llm_data[field]
        except Exception as e:
            logging.error(f"LLM fallback failed: {str(e)}")
    
    # Ensure all fields have values (even if Not Available)
    for field in regex_fields:
        if regex_fields[field] is None:
            regex_fields[field] = "Not Available"
    
    return regex_fields

def similar(a, b, threshold=0.7):
    """Fuzzy string matching for OCR corrections"""
    from difflib import SequenceMatcher
    return SequenceMatcher(None, a, b).ratio() >= threshold

def merge_results(nlp_data, template_data, priority_fields):
    """Combine NLP and template results with priority"""
    merged = nlp_data.copy()
    for field in priority_fields:
        if template_data.get(field, "Unknown") != "Unknown":
            merged[field] = {
                'value': template_data[field],
                'source': 'template',
                'confidence': 0.8  # Template matches get higher confidence
            }
    return merged

def extract_additional_info(text, structured_data):
    """Remove known fields from text for additional info"""
    patterns = [re.escape(v['value']) for v in structured_data.values() 
               if v['value'] != "Unknown"]
    return re.sub(r'|'.join(patterns), '', text, flags=re.IGNORECASE).strip()

def apply_template_fallback(image, existing_data):
    # Define template regions and expected fields
    field_regions = {
        "document_date": (0.1, 0.05, 0.3, 0.08),  # x1, y1, x2, y2 as percentages
        "mammo_type": (0.1, 0.15, 0.4, 0.18),
        "birads_score": (0.6, 0.4, 0.8, 0.45)
    }
    
    h, w = image.shape[:2]
    results = {}
    
    for field, (x1_pct, y1_pct, x2_pct, y2_pct) in field_regions.items():
        # Extract ROI based on template positions
        x1 = int(w * x1_pct)
        y1 = int(h * y1_pct)
        x2 = int(w * x2_pct)
        y2 = int(h * y2_pct)
        roi = image[y1:y2, x1:x2]
        
        # OCR the specific region
        ocr_data = pytesseract.image_to_string(
            roi,
            config='--psm 7 -c tessedit_char_whitelist="0123456789-ABCDEFGHIJKLMNOPQRSTUVWXYZ "'
        )
        results[field] = ocr_data.strip()
    
    # Merge template results with existing data
    if results.get('document_date'):
        existing_data['dates'].append({
            "type": "template_date",
            "value": results['document_date'],
            "confidence": 0.9  # High confidence for template matches
        })
    if results.get('mammo_type'):
        existing_data['procedures'].append({
            "type": "template_procedure",
            "value": results['mammo_type'],
            "confidence": 0.85
        })
    
    return existing_data

def convert_to_structured_json(df):
    """Convert dataframe to medical JSON structure without field_type"""
    return {
        "patient": {
            "name": df.get('patient_name', 'Unknown'),
            "date_of_birth": df.get('date_of_birth', 'Unknown')
        },
        "exam": {
            "date": df.get('exam_date', 'Unknown'),
            "type": df.get('document_type', 'Mammogram'),
            "birads": df.get('birads_score', 'Unknown')
        },
        "findings": {
            "primary": df.get('mammogram_results', 'No results'),
            "additional": df.get('additional_information', '')
        }
    }

def extract_with_template_matching(image):
    """Multi-scale template matching with resource-aware parallel processing"""
    # Map field names to template filenames
    field_template_map = {
        'document_date': 'date_template',
        'exam_type': 'exam_type_template', 
        'birads_score': 'birads_template',
        'patient_name': 'patient_template',
        'impressions': 'impressions_template'
    }
    
    # Adjusted parallel processing settings
    max_workers = min(os.cpu_count(), 4)  # Prevent over-subscription
    scales = np.linspace(0.9, 1.1, 3)  # Reduced scale range for efficiency
    results = {k: "Unknown" for k in field_template_map.keys()}
    results['additional_information'] = ""
    warnings = []
    
    try:
        # Convert to grayscale once
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray_image.shape
        
        # Create process pool for template-scale combinations
        with ThreadPoolExecutor(max_workers=os.cpu_count()*2) as executor:
            futures = []
            
            # Generate scaled templates
            scales = np.linspace(0.8, 1.2, 5)  # 5 scales between 80%-120%
            for field, template_key in field_template_map.items():
                template = TEMPLATES.get(template_key)
                if template is None:
                    warnings.append(f"Template for '{field}' not found")
                    continue
                
                # Submit all scale variations
                for scale in scales:
                    futures.append(
                        executor.submit(
                            process_template_scale,
                            gray_image.copy(),
                            template,
                            field,
                            scale
                        )
                    )

            # Process results as they complete
            best_matches = {}
            for future in concurrent.futures.as_completed(futures):
                field, value, confidence, scale, (x, y) = future.result()
                
                # Track best match per field
                if confidence > best_matches.get(field, (0,))[0]:
                    best_matches[field] = (confidence, scale, x, y, value)

            # Extract text from best matches
            mask = np.zeros_like(gray_image)
            for field, (conf, scale, x, y, val) in best_matches.items():
                if conf < 0.7:  # Final confidence threshold
                    warnings.append(f"Low confidence ({conf:.0%}) for {field}")
                    continue
                
                # Get original template dimensions
                orig_h, orig_w = TEMPLATES[field_template_map[field]].shape
                
                # Calculate ROI based on original size
                roi_x1 = x + int(orig_w * scale)
                roi_y1 = y
                roi_x2 = roi_x1 + 400  # Fixed width for text extraction
                roi_y2 = roi_y1 + int(orig_h * scale) + 50
                
                # Ensure coordinates are within image bounds
                roi = gray_image[
                    max(0, roi_y1):min(h, roi_y2),
                    max(0, roi_x1):min(w, roi_x2)
                ]
                
                # OCR the region
                text = pytesseract.image_to_string(roi, config='--psm 6').strip()
                results[field] = ' '.join(text.split()) or "Unknown"
                
                # Add to mask
                cv2.rectangle(mask, (x, y), 
                             (x + int(orig_w * scale), y + int(orig_h * scale)),
                             (255,255,255), -1)
    except Exception as e:
        logging.error(f"Error processing templates: {str(e)}")
        warnings.append(f"Template processing error: {str(e)}")
