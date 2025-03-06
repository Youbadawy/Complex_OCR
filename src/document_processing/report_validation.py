"""
Validation module for Medical Report Processor application.
This module provides functionality to validate and cross-check extracted report data.
"""

import re
import logging
import json
from typing import Dict, Any, List, Optional, Union, Tuple
import datetime

# Setup logger
logger = logging.getLogger(__name__)

# Import LLM extraction functionality if available
try:
    from .llm_extraction import extract_with_llm, enhance_extraction_with_llm
    LLM_AVAILABLE = True
except ImportError:
    logger.warning("LLM extraction module not available")
    LLM_AVAILABLE = False
    
# Import the BIRADS extraction function
try:
    from .text_analysis import extract_birads_score
except ImportError:
    logger.warning("Could not import extract_birads_score")
    def extract_birads_score(text):
        return {'value': '', 'confidence': 0.0}

class ValidationError(Exception):
    """Base exception for validation errors"""
    pass

class InconsistentDataError(ValidationError):
    """Exception raised when data is inconsistent"""
    pass

class MissingDataError(ValidationError):
    """Exception raised when required data is missing"""
    pass

def is_valid_date(date_str: str) -> bool:
    """
    Check if a string is a valid date.
    
    Args:
        date_str: Date string to validate
        
    Returns:
        Boolean indicating if date is valid
    """
    if not date_str or date_str == "Not Available":
        return False
        
    # Check for YYYY-MM-DD format
    if re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
        try:
            year, month, day = map(int, date_str.split('-'))
            datetime.date(year, month, day)
            return True
        except ValueError:
            return False
            
    return False

def is_valid_birads(birads_str: str) -> bool:
    """
    Check if a string is a valid BIRADS score.
    
    Args:
        birads_str: BIRADS string to validate
        
    Returns:
        Boolean indicating if BIRADS score is valid
    """
    if not birads_str or birads_str == "Not Available":
        return False
        
    # Check for BIRADS X format
    if re.match(r'^BIRADS\s*[0-6][a-c]?$', birads_str, re.IGNORECASE):
        return True
        
    return False

def is_valid_exam_type(exam_type: str) -> bool:
    """
    Check if a string is a valid exam type.
    
    Args:
        exam_type: Exam type string to validate
        
    Returns:
        Boolean indicating if exam type is valid
    """
    if not exam_type or exam_type == "Not Available":
        return False
        
    valid_types = ["MAMMOGRAM", "ULTRASOUND", "MRI", "TOMOSYNTHESIS", "BIOPSY"]
    
    for valid_type in valid_types:
        if valid_type in exam_type.upper():
            return True
            
    return False

def normalize_date(date_str: str) -> str:
    """
    Normalize date to YYYY-MM-DD format.
    
    Args:
        date_str: Date string to normalize
        
    Returns:
        Normalized date string
    """
    if not date_str or date_str == "Not Available":
        return date_str
        
    # Already in YYYY-MM-DD format
    if re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
        return date_str
        
    # Try to parse MM/DD/YYYY format
    if re.match(r'^\d{1,2}/\d{1,2}/\d{4}$', date_str):
        month, day, year = map(int, date_str.split('/'))
        return f"{year:04d}-{month:02d}-{day:02d}"
        
    # Try to parse DD/MM/YYYY format
    if re.match(r'^\d{1,2}/\d{1,2}/\d{4}$', date_str):
        day, month, year = map(int, date_str.split('/'))
        return f"{year:04d}-{month:02d}-{day:02d}"
        
    # Try to parse Month DD, YYYY format
    match = re.match(r'^([A-Za-z]{3,9})\s+(\d{1,2}),?\s+(\d{4})$', date_str)
    if match:
        month_str, day, year = match.groups()
        month_map = {
            "january": 1, "february": 2, "march": 3, "april": 4,
            "may": 5, "june": 6, "july": 7, "august": 8,
            "september": 9, "october": 10, "november": 11, "december": 12,
            "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
            "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12
        }
        month = month_map.get(month_str.lower(), 1)
        return f"{int(year):04d}-{month:02d}-{int(day):02d}"
        
    # Return original if we can't normalize
    return date_str

def normalize_birads(birads_str: str) -> str:
    """
    Normalize BIRADS score to standard format.
    
    Args:
        birads_str: BIRADS string to normalize
        
    Returns:
        Normalized BIRADS string
    """
    if not birads_str or birads_str == "Not Available":
        return birads_str
        
    # Extract BIRADS number and optional letter
    match = re.search(r'[^0-6]*([0-6])([a-c])?', birads_str, re.IGNORECASE)
    if match:
        number = match.group(1)
        letter = match.group(2) or ""
        return f"BIRADS {number}{letter.lower()}"
        
    # Return original if we can't normalize
    return birads_str

def normalize_exam_type(exam_type: str) -> str:
    """
    Normalize exam type to standard format.
    
    Args:
        exam_type: Exam type string to normalize
        
    Returns:
        Normalized exam type string
    """
    if not exam_type or exam_type == "Not Available":
        return exam_type
        
    # Map common variations to standard types
    type_map = {
        "MAMMOGRAM": ["MAMMOGRAPHY", "MAMMO", "BILATERAL MAMMOGRAM", "SCREENING MAMMOGRAM"],
        "ULTRASOUND": ["US", "SONOGRAPHY", "SONOGRAM", "BREAST US"],
        "MRI": ["MAGNETIC RESONANCE IMAGING", "MR"],
        "TOMOSYNTHESIS": ["TOMO", "3D MAMMOGRAM", "3D MAMMO", "3D"],
        "BIOPSY": ["CORE BIOPSY", "NEEDLE BIOPSY", "FNA"]
    }
    
    exam_upper = exam_type.upper()
    
    for standard_type, variations in type_map.items():
        if standard_type in exam_upper:
            return standard_type
            
        for variation in variations:
            if variation in exam_upper:
                return standard_type
                
    # Return original if we can't normalize
    return exam_type

def check_birads_impression_consistency(birads: str, impression: str) -> bool:
    """
    Check if BIRADS score is consistent with impression text.
    
    Args:
        birads: BIRADS score
        impression: Impression text
        
    Returns:
        Boolean indicating if BIRADS is consistent with impression
    """
    if not birads or birads == "Not Available" or not impression:
        return True  # Can't check consistency
        
    birads_value = re.search(r'[^0-6]*([0-6])', birads)
    if not birads_value:
        return True  # Can't extract BIRADS number
        
    birads_num = birads_value.group(1)
    
    # Map BIRADS scores to keywords
    consistency_map = {
        "0": ["incomplete", "additional", "additional imaging", "additional evaluation"],
        "1": ["negative", "normal", "no evidence", "unremarkable"],
        "2": ["benign", "no suspicious", "typical", "usual", "unchanged", "stable", "no concern"],
        "3": ["probably benign", "likely benign", "short-term follow", "follow up in 6"],
        "4": ["suspicious", "suspicious for malignancy", "biopsy", "recommend biopsy"],
        "5": ["highly suspicious", "highly suggestive", "malignancy", "consider biopsy"],
        "6": ["known malignancy", "biopsy proven", "proven malignancy"]
    }
    
    # Check if impression contains keywords consistent with BIRADS
    if any(keyword in impression.lower() for keyword in consistency_map.get(birads_num, [])):
        return True
    
    # Check if impression contains contradictory keywords
    contradictory_scores = {k: v for k, v in consistency_map.items() if k != birads_num}
    for score, keywords in contradictory_scores.items():
        if any(keyword in impression.lower() for keyword in keywords):
            contradictions = [kw for kw in keywords if kw in impression.lower()]
            logger.warning(f"BIRADS {birads_num} contradicts impression: {contradictions}")
            return False
            
    # Default to True if no clear contradiction
    return True

def extract_birads_from_impression(impression: str) -> str:
    """
    Extract BIRADS score from impression text when missing.
    
    Args:
        impression: Impression text
        
    Returns:
        Extracted BIRADS score or empty string
    """
    if not impression:
        return ""
        
    # First try direct extraction
    birads_result = extract_birads_score(impression)
    if birads_result['value']:
        return birads_result['value']
        
    # Then try keyword-based extraction
    keyword_map = {
        "negative": "BIRADS 1",
        "normal": "BIRADS 1",
        "no evidence of malignancy": "BIRADS 1",
        "benign": "BIRADS 2", 
        "stable": "BIRADS 2",
        "no suspicious": "BIRADS 2",
        "probably benign": "BIRADS 3",
        "likely benign": "BIRADS 3",
        "suspicious": "BIRADS 4",
        "recommend biopsy": "BIRADS 4",
        "highly suspicious": "BIRADS 5",
        "highly suggestive": "BIRADS 5",
        "known malignancy": "BIRADS 6",
        "biopsy proven": "BIRADS 6"
    }
    
    for keyword, birads in keyword_map.items():
        if keyword in impression.lower():
            logger.info(f"Inferred {birads} from impression text containing '{keyword}'")
            return birads
            
    return ""

def resolve_field_conflicts(
    field_name: str, 
    values: List[str], 
    confidence_scores: Optional[List[float]] = None
) -> str:
    """
    Resolve conflicts between different values for the same field.
    
    Args:
        field_name: Name of the field
        values: List of different values for the field
        confidence_scores: Optional list of confidence scores for each value
        
    Returns:
        Resolved field value
    """
    # Filter out empty values and "Not Available"
    filtered_values = [v for v in values if v and v != "Not Available"]
    
    if not filtered_values:
        return "Not Available"
        
    if len(filtered_values) == 1:
        return filtered_values[0]
        
    # If we have confidence scores, use the value with highest confidence
    if confidence_scores and len(confidence_scores) == len(values):
        valid_values = [(v, c) for v, c in zip(values, confidence_scores) if v and v != "Not Available"]
        if valid_values:
            best_value, best_score = max(valid_values, key=lambda x: x[1])
            logger.info(f"Resolved {field_name} conflict using confidence scores: {best_value} (confidence: {best_score})")
            return best_value
    
    # Field-specific resolution logic
    if field_name == "birads_score":
        # Normalize all BIRADS values and check if they agree after normalization
        normalized_values = [normalize_birads(v) for v in filtered_values]
        if len(set(normalized_values)) == 1:
            return normalized_values[0]
            
        # Prefer higher BIRADS if in conflict (conservative approach)
        birads_numbers = []
        for value in filtered_values:
            match = re.search(r'[^0-6]*([0-6])', value)
            if match:
                birads_numbers.append(int(match.group(1)))
                
        if birads_numbers:
            highest_birads = max(birads_numbers)
            for value in filtered_values:
                if str(highest_birads) in value:
                    logger.info(f"Resolved BIRADS conflict by choosing highest value: {value}")
                    return value
                    
    elif field_name == "exam_date":
        # Try to normalize dates and see if they agree
        normalized_dates = [normalize_date(v) for v in filtered_values]
        if len(set(normalized_dates)) == 1:
            return normalized_dates[0]
            
        # Prefer most recent date if multiple valid dates
        valid_dates = []
        for date_str in normalized_dates:
            if re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
                try:
                    year, month, day = map(int, date_str.split('-'))
                    valid_dates.append((date_str, datetime.date(year, month, day)))
                except ValueError:
                    continue
                    
        if valid_dates:
            most_recent = max(valid_dates, key=lambda x: x[1])
            logger.info(f"Resolved date conflict by choosing most recent: {most_recent[0]}")
            return most_recent[0]
    
    # Default: return the longest value as it might contain more information
    longest_value = max(filtered_values, key=len)
    logger.info(f"Resolved {field_name} conflict by choosing longest value: {longest_value}")
    return longest_value

def fix_impression_duplication(report_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fix duplicate impression content in findings and other fields.
    
    Args:
        report_data: Report data with possible impression duplication
        
    Returns:
        Report data with fixed impression fields
    """
    fixed_data = report_data.copy()
    
    # Check if impression appears in findings
    if fixed_data.get('impression') and fixed_data.get('findings'):
        impression = fixed_data['impression']
        findings = fixed_data['findings']
        
        # If impression text is duplicated in findings, remove it from findings
        if impression in findings:
            findings = findings.replace(impression, "").strip()
            if findings:
                fixed_data['findings'] = findings
            else:
                fixed_data['findings'] = "See impression section."
                
    # Check if recommendation appears in impression
    if fixed_data.get('impression') and fixed_data.get('recommendation'):
        impression = fixed_data['impression']
        recommendation = fixed_data['recommendation']
        
        # If recommendation text is duplicated in impression, remove it from impression
        if recommendation in impression:
            impression = impression.replace(recommendation, "").strip()
            if impression:
                fixed_data['impression'] = impression
                
    return fixed_data

def augment_missing_fields(report_data: Dict[str, Any], raw_text: str) -> Dict[str, Any]:
    """
    Try to fill in missing fields using available information.
    
    Args:
        report_data: Report data with possible missing fields
        raw_text: Original raw text
        
    Returns:
        Augmented report data
    """
    augmented_data = report_data.copy()
    
    # Try to extract BIRADS score from impression if missing
    if augmented_data.get('birads_score') in ["", "Not Available"] and augmented_data.get('impression'):
        birads = extract_birads_from_impression(augmented_data['impression'])
        if birads:
            augmented_data['birads_score'] = birads
            logger.info(f"Extracted BIRADS {birads} from impression")
            
    # Try to extract BIRADS score from raw text if still missing
    if augmented_data.get('birads_score') in ["", "Not Available"]:
        birads_result = extract_birads_score(raw_text)
        if birads_result['value'] and birads_result['confidence'] >= 0.7:
            augmented_data['birads_score'] = birads_result['value']
            logger.info(f"Extracted BIRADS {birads_result['value']} from raw text")
            
    # Use LLM as a last resort if available
    if LLM_AVAILABLE and any(augmented_data.get(field) in ["", "Not Available"] for field in ['birads_score', 'impression', 'findings']):
        try:
            augmented_data = enhance_extraction_with_llm(augmented_data, raw_text)
            logger.info("Enhanced extraction with LLM")
        except Exception as e:
            logger.error(f"LLM enhancement failed: {str(e)}")
            
    return augmented_data

def validate_report_data(report_data: Dict[str, Any], raw_text: str) -> Dict[str, Any]:
    """
    Validate and clean report data.
    
    Args:
        report_data: Report data to validate
        raw_text: Original raw text
        
    Returns:
        Validated and cleaned report data
        
    Raises:
        ValidationError: If validation fails and cannot be fixed
    """
    logger.info("Validating report data")
    
    # Create a copy of the data to avoid modifying the original
    validated_data = report_data.copy()
    
    # Step 1: Normalize fields
    if 'exam_date' in validated_data:
        validated_data['exam_date'] = normalize_date(validated_data['exam_date'])
        
    if 'birads_score' in validated_data:
        validated_data['birads_score'] = normalize_birads(validated_data['birads_score'])
        
    if 'exam_type' in validated_data:
        validated_data['exam_type'] = normalize_exam_type(validated_data['exam_type'])
        
    # Step 2: Fix impression duplication
    validated_data = fix_impression_duplication(validated_data)
    
    # Step 3: Check for birads-impression consistency
    if ('birads_score' in validated_data and 'impression' in validated_data and 
            validated_data['birads_score'] not in ["", "Not Available"] and 
            validated_data['impression'] not in ["", "Not Available"]):
        if not check_birads_impression_consistency(validated_data['birads_score'], validated_data['impression']):
            logger.warning(f"BIRADS score {validated_data['birads_score']} inconsistent with impression")
            
            # Extract BIRADS from impression as a second opinion
            impression_birads = extract_birads_from_impression(validated_data['impression'])
            
            if impression_birads and impression_birads != validated_data['birads_score']:
                # Resolve conflict
                logger.info(f"BIRADS conflict: {validated_data['birads_score']} vs {impression_birads}")
                validated_data['birads_score'] = resolve_field_conflicts(
                    'birads_score', 
                    [validated_data['birads_score'], impression_birads],
                    [0.8, 0.9]  # Slightly prefer impression-derived BIRADS
                )
    
    # Step 4: Augment missing fields
    validated_data = augment_missing_fields(validated_data, raw_text)
    
    logger.info("Report data validation complete")
    return validated_data

def extract_and_validate_report(raw_text: str) -> Dict[str, Any]:
    """
    Extract and validate report data from raw text.
    
    This function combines extraction and validation in a single step,
    using traditional extraction methods first, then LLM if needed.
    
    Args:
        raw_text: Raw OCR text
        
    Returns:
        Validated report data
    """
    from .text_analysis import process_document_text
    
    try:
        # First use traditional extraction methods
        logger.info("Extracting report data with traditional methods")
        extracted_data = process_document_text(raw_text)
        
        # Validate and enhance the extracted data
        logger.info("Validating and enhancing extracted data")
        validated_data = validate_report_data(extracted_data, raw_text)
        
        # If critical fields are still missing, try LLM extraction
        critical_fields = ["exam_date", "exam_type", "birads_score", "impression"]
        missing_fields = [field for field in critical_fields 
                         if field not in validated_data or 
                         validated_data[field] in ["", "Not Available"]]
        
        if missing_fields and LLM_AVAILABLE:
            logger.info(f"Critical fields still missing: {missing_fields}")
            logger.info("Trying LLM extraction")
            
            try:
                llm_data = extract_with_llm(raw_text, "full")
                
                # Merge LLM-extracted data with validated data, prioritizing LLM for missing fields
                for field in missing_fields:
                    if field in llm_data and llm_data[field]:
                        validated_data[field] = llm_data[field]
                        logger.info(f"LLM extracted {field}: {llm_data[field]}")
                
                # Re-validate the merged data
                validated_data = validate_report_data(validated_data, raw_text)
                
            except Exception as e:
                logger.error(f"LLM extraction failed: {str(e)}")
        
        return validated_data
        
    except Exception as e:
        logger.error(f"Error in extract_and_validate_report: {str(e)}")
        raise 