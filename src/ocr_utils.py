import cv2
import pytesseract
import re
import pandas as pd
from pytesseract import Output
import os
import numpy as np
from langdetect import detect, LangDetectException
from transformers import MarianMTModel, MarianTokenizer, pipeline as tf_pipeline
import torch
from spellchecker import SpellChecker
import logging
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures

def preprocess_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Reduce noise with Gaussian blur
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Adaptive thresholding for varied lighting conditions
    thresholded = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )
    
    # Morphological opening to remove small artifacts
    kernel = np.ones((2,2), np.uint8)
    processed = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel)
    
    return processed

def extract_text_tesseract(image):
    data = pytesseract.image_to_data(
        image,
        output_type=Output.DICT,
        lang='fra+eng',
        config=f'--psm {os.getenv("TESS_PSM", "6")} '
               '-c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-:.()%éèàçêâôûùîÉÈÀÇÊÂÔÛÙÎ" '
               '--oem 3'
    )
    return {
        'text': data['text'],
        'confidence': data['conf']
    }

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

def extract_fields_from_text(text, nlp_pipeline, image, max_length=510):
    """Always returns (structured_data, warnings) tuple"""
    structured_data = default_structured_output()
    warnings = []
    
    try:
        # Clean and validate input text
        clean_text = ' '.join(str(text).strip().split())[:max_length]
        if not clean_text or len(clean_text) < 10:
            return structured_data, warnings
        
        # Detect language with fallback to English
        try:
            lang = detect(clean_text)
        except LangDetectException:
            lang = 'en'
        
        # Load translation pipeline if needed
        if lang == 'fr':
            translator = tf_pipeline(
                "translation_fr_to_en",
                model="Helsinki-NLP/opus-mt-fr-en",
                device=0 if torch.cuda.is_available() else -1
            )
            translated = translator(clean_text, max_length=512)[0]['translation_text']
            clean_text = translated

        # Process with medical NLP
        entities = nlp_pipeline(clean_text)
        
        # Structure entities into medical categories
        structured_data = {
            "patient_info": [],
            "dates": [],
            "procedures": [],
            "findings": [],
            "birads_scores": [],
            "recommendations": []
        }

        current_procedure = None
        for entity in entities:
            ent_text = entity['word']
            ent_type = entity['entity']
            
            # Map entity types to medical categories
            if ent_type in ['B-PER', 'I-PER']:
                structured_data["patient_info"].append({
                    "type": "patient_name",
                    "value": ent_text,
                    "confidence": entity['score']
                })
            elif ent_type in ['B-DATE', 'I-DATE']:
                structured_data["dates"].append({
                    "type": "exam_date",
                    "value": ent_text,
                    "confidence": entity['score']
                })
            elif "PROCEDURE" in ent_type:
                current_procedure = ent_text
                structured_data["procedures"].append({
                    "type": "imaging_type",
                    "value": ent_text,
                    "confidence": entity['score']
                })
            elif "FINDING" in ent_type and current_procedure:
                structured_data["findings"].append({
                    "procedure": current_procedure,
                    "description": ent_text,
                    "confidence": entity['score']
                })
            elif "BIRADS" in ent_text.upper():
                structured_data["birads_scores"].append({
                    "type": "birads",
                    "value": ent_text,
                    "confidence": entity['score']
                })
            elif "RECOMMEND" in ent_type:
                structured_data["recommendations"].append({
                    "type": "follow_up",
                    "value": ent_text,
                    "confidence": entity['score']
                })

        # Post-process with regex patterns
        date_matches = re.findall(r'\b\d{4}-\d{2}-\d{2}\b', clean_text)
        structured_data["dates"].extend([{
            "type": "regex_date",
            "value": date,
            "confidence": 1.0
        } for date in date_matches])

        # Get OCR data with confidence scores
        ocr_data = extract_text_tesseract(image)
        ocr_words = ocr_data['text']
        ocr_confs = [c/100 for c in ocr_data['confidence'] if c != -1]
        
        # Calculate field confidence from OCR
        field_confidences = {}
        for idx, entity in enumerate(entities):
            field_type = entity['entity'].split('-')[-1]
            word = entity['word']
            
            # Find matching OCR words with confidence
            matches = [ocr_confs[i] for i, w in enumerate(ocr_words) 
                      if similar(w, word, threshold=0.7)]
            
            if matches:
                field_conf = sum(matches)/len(matches)
                field_confidences.setdefault(field_type, []).append(field_conf)
        
        # Check for low confidence fields
        low_conf_fields = [
            field for field, confs in field_confidences.items() 
            if sum(confs)/len(confs) < 0.5
        ]
        
        # Fallback to template matching for low confidence fields
        if low_conf_fields:
            template_results, template_warnings = extract_with_template_matching(image)
            structured_data = merge_results(
                structured_data, 
                template_results,
                priority_fields=low_conf_fields
            )
            warnings.extend(template_warnings)
        
        # Add additional information from remaining text
        structured_data['additional_information'] = extract_additional_info(
            clean_text, 
            structured_data
        )
        
        # Return both data and warnings
        return structured_data, warnings
    
    except Exception as e:
        logging.error(f"Field extraction failed: {str(e)}")
        return structured_data, ["Field extraction failed - using template fallback"]

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
