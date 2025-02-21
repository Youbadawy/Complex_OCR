import cv2
import pytesseract
import re
import pandas as pd
from pytesseract import Output
import os
import numpy as np
import dotenv
import requests
import json
from typing import Dict, Any
from langdetect import detect, LangDetectException
from transformers import MarianMTModel, MarianTokenizer, pipeline as tf_pipeline
import torch
from spellchecker import SpellChecker
import logging
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures

def preprocess_image(image):
    """Preprocess image with deskewing and adaptive thresholding"""
    # Convert to grayscale and blur
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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
    return binary

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
            timeout=30  # Set 30 second timeout per PSM
            config=f'--psm {psm} '
                   '-c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-:.()%éèàçêâôûùîÉÈÀÇÊÂÔÛÙÎ" '
                   '--oem 3'
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
    
    # Select best result using LLM or confidence
    best_result = select_best_ocr_result(all_results)
    return {
        'text': best_result['text'],
        'confidence': best_result['confidence'],
        'psm': best_result['psm']
    }

def select_best_ocr_result(results):
    """Select best OCR result using LLM or confidence scores"""
    try:
        # Try LLM selection first
        try:
            text_options = "\n\n".join([f"Option {i+1} (PSM {r['psm']}): {' '.join(r['text'])}" 
                                      for i, r in enumerate(results)])
            
            llm_response = parse_text_with_llm(
            f"Which OCR option is most coherent? Return only the number:\n{text_options}"
            )
            
            if 'structured_data' in llm_response and 'choice' in llm_response['structured_data']:
                chosen_idx = int(llm_response['structured_data']['choice']) - 1
                return results[chosen_idx]
            
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            logging.warning(f"LLM selection timeout: {str(e)}")
        except KeyError as e:
            logging.warning(f"Invalid LLM response format: {str(e)}")
    except Exception as e:
        logging.warning(f"LLM selection failed: {str(e)}")
    
    # Fallback to highest average confidence
    return max(results, key=lambda x: x['avg_conf'])

def parse_extracted_text(ocr_result):
    raw_text = ' '.join([t for t, c in zip(ocr_result['text'], ocr_result['confidence']) if c != -1])
    confidences = [c/100 for t, c in zip(ocr_result['text'], ocr_result['confidence']) if c != -1]
    
    # Initialize spell checker with medical dictionary
    spell = SpellChecker()
    spell.word_frequency.load_text_file('medical_terms.txt')
    
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

def parse_text_with_llm(text: str) -> Dict[str, Any]:
    """Enhance OCR text parsing using Groq's LLM API"""
    try:
        dotenv.load_dotenv()
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            logging.error("Groq API key not found in .env file")
            raise OCRError("API configuration error")

    prompt = f"""Analyze this medical report text and extract structured data. 
    Correct any OCR errors, especially fused words. Return JSON with these fields:
    - patient_name (title case)
    - exam_date (YYYY-MM-DD)
    - exam_type (uppercase)
    - birads_right (0-6)
    - birads_left (0-6)
    - clinical_history (concise summary)
    - findings (key observations)
    - follow_up_recommendation (action items)
    
    Text: {text[:3000]}..."""  # Truncate to stay under token limits

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "mixtral-8x7b-32768",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 1000
    }

    try:
        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=15  # Increased timeout
            )
            response.raise_for_status()
            
        except requests.exceptions.Timeout:
            logging.error("LLM API timeout")
            raise OCRError("AI processing timed out")
        except requests.exceptions.HTTPError as e:
            logging.error(f"LLM API error: {e.response.status_code}")
            raise OCRError("AI service unavailable")
        
        # Extract JSON from response
        result = json.loads(response.json()['choices'][0]['message']['content'])
        return result.get("structured_data", {})
        
    except (requests.exceptions.RequestException, json.JSONDecodeError, KeyError) as e:
        logging.warning(f"LLM API error: {str(e)}")
        return {}

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
        # Patient name fallback
        if 'patient_name' in missing_fields:
            if match := re.search(r'patient name[:\s]+([a-z\s]+)', text, re.IGNORECASE):
                fields['patient_name'] = match.group(1).strip().title()
        
        # Exam date fallback
        if 'exam_date' in missing_fields:
            if match := re.search(r'exam date[:\s]+([0-9]{4}-[0-9]{2}-[0-9]{2}|[0-9]{2}/[0-9]{2}/[0-9]{4})', text, re.IGNORECASE):
                fields['exam_date'] = match.group(1).strip()
        
        # BIRADS scores fallback
        if 'birads_right' in missing_fields:
            if match := re.search(r'birads right[:\s]+([0-6])', text, re.IGNORECASE):
                fields['birads_right'] = match.group(1)
        if 'birads_left' in missing_fields:
            if match := re.search(r'birads left[:\s]+([0-6])', text, re.IGNORECASE):
                fields['birads_left'] = match.group(1)

        # Clinical sections fallback
        section_mappings = {
            'impressions': r'(impressions|conclusion)[:\s]+(.*?)\s+(findings|recommendation)',
            'findings': r'findings[:\s]+(.*?)\s+(recommendation|follow-up)',
            'follow-up_recommendation': r'(follow-up|recommendation)[:\s]+(.*?)\s+(signed|printed|end of document)',
            'clinical_history': r'clinical history[:\s]+(.*?)\s+(impressions|findings|recommendation)',
            'exam_type': r'exam type[:\s]+([a-z\s]+)',
            'document_date': r'(document date|date of exam)[:\s]+([0-9]{4}-[0-9]{2}-[0-9]{2})'
        }

        for field, pattern in section_mappings.items():
            if field in missing_fields:
                if match := re.search(pattern, text, re.IGNORECASE | re.DOTALL):
                    fields[field] = match.group(2 if field == 'document_date' else 1).strip()
                    if field == 'exam_type':
                        fields[field] = fields[field].upper()
    
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
    """Multi-scale template matching with parallel processing"""
    # Map field names to template filenames
    field_template_map = {
        'document_date': 'date_template',
        'exam_type': 'exam_type_template', 
        'birads_score': 'birads_template',
        'patient_name': 'patient_template',
        'impressions': 'impressions_template'
    }
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
