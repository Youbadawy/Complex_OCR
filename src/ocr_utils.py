import cv2
import pytesseract
import re
import asyncio
import aiohttp
from tenacity import AsyncRetrying
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
    """Hybrid OCR pipeline with text validation"""
    # Preprocessing
    enhanced = cv2.convertScaleAbs(image, alpha=1.5, beta=0)
    processed = preprocess_image(enhanced)
    
    # Multi-engine OCR
    paddle_text = extract_text_paddle(processed)
    tesseract_text = extract_text_tesseract(processed)
    
    # Validate and combine
    validated = validate_results({
        "paddle": paddle_text,
        "tesseract": tesseract_text
    })
    
    return validated

def validate_results(texts: dict) -> str:
    """Validate OCR results against medical terms"""
    corrections = {
        r"\bIMPRessiON\b": "IMPRESSION",
        r"\baimost\b": "almost",
        r"\bMamm0gram\b": "Mammogram"
    }
    
    best_text = max(texts.values(), key=lambda t: medical_term_score(t))
    
    # Apply regex corrections
    for pattern, replacement in corrections.items():
        best_text = re.sub(pattern, replacement, best_text, flags=re.IGNORECASE)
        
    return best_text

def medical_term_score(text: str) -> int:
    """Score text based on medical term presence"""
    return sum(
        1 for term in MEDICAL_TERMS
        if re.search(rf"\b{term}\b", text, re.IGNORECASE)
    )

def extract_text_paddle(image: np.ndarray) -> str:
    """Extract text using PaddleOCR"""
    try:
        from paddleocr import PaddleOCR
        ocr = PaddleOCR(use_angle_cls=True, lang='en')
        result = ocr.ocr(image, cls=True)
        if result and result[0]:
            return ' '.join([line[1][0] for line in result[0]])
        return ''
    except Exception as e:
        logging.error(f"PaddleOCR failed: {str(e)}")
        return ''

def preprocess_image(image, apply_sharpen=True):
    """Preprocess image with adaptive noise handling and GPU acceleration"""
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        gpu_img = cv2.cuda_GpuMat()
        gpu_img.upload(image)
        # CUDA-accelerated operations
        gray = cv2.cuda.cvtColor(gpu_img, cv2.COLOR_BGR2GRAY)
        contrast = cv2.cuda.mean(cv2.cuda.Laplacian(gray, cv2.CV_64F))[0]
        result = gray.download()
    else:
        # CPU fallback
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contrast = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Adaptive noise reduction for low-contrast images
    if contrast < 50:  # Indicates noisy/poor quality image
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
    
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Deskewing logic
    height, width = blur.shape
    edges = cv2.Canny(blur, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100,
                          minLineLength=100, maxLineGap=10)
    
    angles = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            angles.append(angle)
            
        median_angle = np.median(angles)
        rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), median_angle, 1)
        deskewed = cv2.warpAffine(blur, rotation_matrix, (width, height),
                                 borderMode=cv2.BORDER_REPLICATE)
    else:
        deskewed = blur
    
    # Adaptive thresholding
    binary = cv2.adaptiveThreshold(deskewed, 255, 
                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 11, 2)
    
    # Optional sharpening for faded text
    if apply_sharpen:
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        binary = cv2.filter2D(binary, -1, kernel)
        
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
    
    # Enhanced medical spell checking
    spell = SpellChecker(language=None)
    spell.word_frequency.load_text_file('medical_terms.txt')
    spell.word_frequency.load_text_file('french_medical_terms.txt')
    
    # Add common OCR error mappings
    ocr_corrections = {
        'aimost': 'almost', 'IMPRessiON': 'IMPRESSION',
        'IMPLANTSE': 'IMPLANTS', 'HISTORy': 'HISTORY'
    }
    
    # Context-aware correction
    corrected_words = []
    for word in raw_text.split():
        # Apply direct substitutions first
        word = ocr_corrections.get(word, word)
        
        # Medical term validation
        if word.lower() not in spell and len(word) > 3:
            candidates = spell.candidates(word)
            if candidates:
                best = max(candidates, key=lambda x: spell.word_usage_frequency(x))
                word = best
                
        corrected_words.append(word)
    
    # Add structured field validation
    structured_text = '\n'.join(corrected_words)
    structured_text = re.sub(
        r'(?i)(BIRADS|IMPRESSION|MAMMOGRAM|HISTORY|ULTRASOUND):',
        lambda m: f"\n{m.group(1).upper()}:\n",
        structured_text
    )
    
    # Clean and structure text from complex layouts
    lines = []
    current_line = []
    for i, word in enumerate(raw_text.split()):
        # Correct spelling
        corrected = spell.correction(word)
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
        r'\b(20\d{2}(?:-(0[1-9]|1[0-2])(?:-(0[1-9]|[12][0-9]|3[01]))?)?)\b'  # YYYY-MM-DD
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
def parse_text_with_llm(text: str) -> Dict[str, Any]:
    """Extract structured fields from OCR text using Groq's LLM API"""
    dotenv.load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        logging.error("Groq API key not found in .env file")
        raise OCRError("API configuration error")

    client = Groq(api_key=api_key)
    
    prompt = f"""You are an expert in medical document processing, specializing in mammogram reports. Extract the following fields:
- Patient name (title case, correct fused words like 'PatientName' to 'Patient Name')
- Exam date (YYYY-MM-DD, correct formats like '23Dec2021')
- BIRADS score for the right breast (0-6)
- BIRADS score for the left breast (0-6)
- Clinical impressions
- Key findings (include measurements and locations)
- Follow-up recommendations

Correct OCR errors, prioritize medical terminology, and handle varied labels. For each field, provide a confidence score (0-1). Return valid JSON with keys: 
patient_name, exam_date, birads_right, birads_left, impressions, findings, 
follow_up_recommendation, and a 'confidence' sub-object with scores for each field.

Text:
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
        required = ['patient_name', 'exam_date', 'birads_right', 'birads_left',
                   'impressions', 'findings', 'follow_up_recommendation', 'confidence']
        if all(k in result for k in required) and isinstance(result['confidence'], dict):
            return result
            
        raise ValueError("Invalid response structure from LLM")
            
    except Exception as e:
        logging.error(f"LLM extraction failed: {str(e)}")
        raise OCRError("Failed to extract fields after 3 attempts") from e

def extract_fields_from_text(text: str, use_llm_fallback: bool = True) -> Dict[str, Any]:
    """Extract fields using LLM first, with regex fallback for missing fields"""
    fields = {
        'patient_name': None,
        'exam_date': None,
        'birads_right': None,
        'birads_left': None,
        'impressions': None,
        'findings': None,
        'follow-up_recommendation': None,
        'document_date': None,
        'exam_type': None,
        'clinical_history': None
    }

    # First try LLM extraction
    llm_fields = {}
    if use_llm_fallback:
        try:
            llm_fields = parse_text_with_llm(text)
            # Map LLM fields to our schema
            fields.update({
                'patient_name': llm_fields.get('patient_name'),
                'exam_date': llm_fields.get('exam_date'),
                'birads_right': llm_fields.get('birads_right'),
                'birads_left': llm_fields.get('birads_left'),
                'impressions': llm_fields.get('impressions'),
                'findings': llm_fields.get('findings'),
                'follow-up_recommendation': llm_fields.get('follow_up_recommendation'),
                'document_date': llm_fields.get('document_date'),
                'exam_type': llm_fields.get('exam_type'),
                'clinical_history': llm_fields.get('clinical_history')
            })
        except Exception as e:
            logging.warning(f"LLM extraction failed: {str(e)}")

    # Regex fallback for any missing critical fields
    missing_fields = [k for k, v in fields.items() if v is None]
    if missing_fields:
        # Enhanced regex patterns with multi-line support and medical context
        patterns = {
            'patient_name': (
                r'(?i)(?:patient|name)(?:\s*(?:name|ID)\b)?[:\s\-*]+'
                r'((?:[A-ZÀ-ÿ][a-zà-ÿ]*-?)+\s*(?:[A-ZÀ-ÿ][a-zà-ÿ]*(?:\s+[A-ZÀ-ÿ][a-zà-ÿ]*)*))'
            ),
            'exam_date': (
                r'(?i)(?:date\s*(?:of\s*)?(?:exam|study)|exam\s*date)[:\s\-*]+'
                r'(\b(?:20\d{2}[-/](?:0[1-9]|1[0-2])[-/](?:0[1-9]|[12][0-9]|3[01])|'
                r'(?:0[1-9]|1[0-2])[-/](?:0[1-9]|[12][0-9]|3[01])[-/]20\d{2})\b)'
            ),
            'birads_right': (
                r'(?i)(?:bi-rads|birads|breast\s+imaging\s+reporting\s+.*?)\s*'
                r'(?:right|rt\.?)\b[\s:\-]*(\d)'
            ),
            'birads_left': (
                r'(?i)(?:bi-rads|birads|breast\s+imaging\s+reporting\s+.*?)\s*'
                r'(?:left|lt\.?)\b[\s:\-]*(\d)'
            ),
            'impressions': (
                r'(?i)(?:impressions?|conclusions?)\b[:\s]*'
                r'((?:.*?(?:\n\s*.*?)*?)(?=\n\s*(?:recommendations?|findings|follow-?up|$))'
            ),
            'clinical_history': (
                r'(?i)(?:clinical\s+history|patient\s+history)\b[:\s]*'
                r'((?:.*?(?:\n\s*.*?)*?)(?=\n\s*(?:exam|findings|impressions|$))'
            ),
            'findings': (
                r'(?i)(?:findings|results)\b[:\s]*'
                r'((?:.*?(?:\n\s*.*?)*?)(?=\n\s*(?:impressions|recommendations?|$))'
            ),
            'follow-up_recommendation': (
                r'(?i)(?:recommendations?|follow-?up)\b[:\s]*'
                r'((?:.*?(?:\n\s*.*?)*?)(?=\n\s*(?:end\s+of\s+report|$)))'
            )
        }

        for field, pattern in patterns.items():
            if field in missing_fields and (match := re.search(pattern, text, re.DOTALL)):
                captured = match.group(1).strip()
                # Post-process based on field type
                if field == 'patient_name':
                    captured = re.sub(r'\s+', ' ', captured).title()
                elif field in ('birads_right', 'birads_left'):
                    captured = max(0, min(int(captured), 6)) if captured.isdigit() else None
                elif field == 'exam_date':
                    captured = re.sub(r'[/]', '-', captured)
                
                if captured:
                    fields[field] = captured

    
    return fields, []

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

            # Process remaining text
            masked_image = cv2.bitwise_and(gray_image, cv2.bitwise_not(mask))
            remaining_text = pytesseract.image_to_string(masked_image)
            results['additional_information'] = clean_additional_text(remaining_text)

    except Exception as e:
        error_msg = f"Template matching failed: {str(e)}"
        logging.error(error_msg, exc_info=True)
        warnings.append(error_msg)
    
    return results, warnings

def process_template_scale(image, template, field, scale):
    """Process a single template at specific scale"""
    try:
        # Scale the template
        scaled_template = cv2.resize(
            template, 
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA
        )
        t_h, t_w = scaled_template.shape
        
        # Skip if template larger than image
        if t_h > image.shape[0] or t_w > image.shape[1]:
            return field, "Unknown", 0.0, scale, (0,0)

        # Match template
        res = cv2.matchTemplate(image, scaled_template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        
        return field, "Unknown", max_val, scale, max_loc
    
    except Exception as e:
        logging.warning(f"Scale {scale} failed for {field}: {str(e)}")
        return field, "Unknown", 0.0, scale, (0,0)

def clean_additional_text(text):
    """Clean and format remaining OCR text"""
    # Remove known fields and noise
    patterns_to_remove = [
        r'Document Date:.*?\n',
        r'BIRADS Score:.*?\n',
        r'Patient Name:.*?\n',
        r'[\x00-\x1F\x7F-\x9F]'  # Remove control characters
    ]
    
    for pattern in patterns_to_remove:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
    return ' '.join(text.split())

def apply_regex_validation(results):
    """Validate and format extracted fields using regex"""
    # Date validation
    date_pattern = r'\b\d{4}-\d{2}-\d{2}\b'
    for field in ['document_date', 'exam_date']:
        match = re.search(date_pattern, results[field])
        if match:
            results[field] = match.group()
            
    # BIRADS score validation
    birads_pattern = r'BIRADS[\s-]*([0-6])'
    if 'birads_score' in results:
        match = re.search(birads_pattern, results['birads_score'], re.IGNORECASE)
        if match:
            results['birads_score'] = f"BIRADS {match.group(1)}"
            
    # Name validation
    name_pattern = r'^[A-Z][a-z]+ [A-Z][a-z]+$'
    if not re.match(name_pattern, results['patient_name']):
        results['patient_name'] = "Unknown"
        
    return results

def extract_findings_text(findings):
    """Convert structured findings data to plain text with error handling"""
    try:
        # Handle stringified JSON from CSV imports
        if isinstance(findings, str):
            try:
                findings = json.loads(findings)
            except json.JSONDecodeError:
                return findings  # Return raw string if not JSON

        # Process list of finding dictionaries
        if isinstance(findings, list):
            descriptions = []
            for item in findings:
                if isinstance(item, dict):
                    desc = item.get('description', '')
                    if desc:  # Only add non-empty descriptions
                        descriptions.append(desc.strip(' .'))
            return ' '.join(descriptions) if descriptions else ''
        
        # Handle single dictionary case
        if isinstance(findings, dict):
            return findings.get('description', '')
        
        return str(findings)
    
    except Exception as e:
        logging.error(f"Findings extraction error: {str(e)}", exc_info=True)
        return ''  # Return empty string on failure

def default_structured_output():
    """Fallback structure for failed extractions"""
    return {
        'birads_score': "Unknown",
        'document_date': "Unknown",
        'document_type': "Unknown",
        'electronically_signed_by': "Unknown",
        'exam_date': "Unknown",
        'impression_result': "Unknown",
        'mammogram_results': "Unknown",
        'patient_history': "Unknown", 
        'patient_name': "Unknown",
        'recommendation': "Unknown",
        'testing_provider': "Unknown",
        'ultrasound_results': "Unknown",
        'additional_information': "",
        'confidence_scores': {'ocr': 0.0, 'nlp': 0.0}
    }

# Add missing dependency installation
def install_dependencies():
    """Ensure required NLP dependencies exist"""
    try:
        import sentencepiece  # Required for translation models
    except ImportError:
        import sys, subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "sentencepiece"])
