"""
Text analysis module for Medical Report Processor application.
This module provides functions for analyzing and extracting information from medical text.
"""

import re
import logging
from typing import Dict, Any, Optional, List, Tuple

# Import functions from pdf_processor.py
from .pdf_processor import (
    extract_sections,
    extract_patient_info,
    extract_date_from_text,
    extract_exam_type
)

# Re-export these functions
__all__ = [
    'extract_sections',
    'extract_patient_info',
    'extract_date_from_text',
    'extract_exam_type',
    'extract_birads_score',
    'extract_provider_info',
    'extract_signed_by',
    'process_document_text',
    'extract_structured_data'
]

def extract_birads_score(text: str) -> Dict[str, Any]:
    """
    Extract BIRADS score from text with enhanced pattern matching.
    
    Args:
        text: Text to extract BIRADS score from
        
    Returns:
        Dictionary with 'value' and 'confidence' keys
    """
    if not text:
        return {'value': '', 'confidence': 0.0}
    
    # Patterns for BIRADS score extraction with different formats
    patterns = [
        # Standard BIRADS format
        (r'(?:BIRADS|BI-RADS)(?:\s+(?:SCORE|CATEGORY|CLASSIFICATION|ASSESSMENT))?(?:\s*[:=]\s*|\s+)([0-6][a-c]?)', 0.95),
        # ACR format
        (r'(?:ACR|Category)(?:\s+(?:SCORE|CATEGORY|CLASSIFICATION|ASSESSMENT))?(?:\s*[:=]\s*|\s+)([0-6][a-c]?)', 0.85),
        # Assessment format
        (r'(?:Assessment|Impression)(?:\s*[:=]\s*|\s+)(?:BIRADS|BI-RADS|ACR)(?:\s+|\s*[:=]\s*)([0-6][a-c]?)', 0.95),
        # Mammogram shows Category X
        (r'(?:shows|indicating|consistent with)(?:\s+)(?:Category|BIRADS|BI-RADS|ACR)(?:\s+|\s*[:=]\s*)([0-6][a-c]?)', 0.85)
    ]
    
    for pattern, confidence in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return {
                'value': f'BIRADS {match.group(1)}',
                'confidence': confidence
            }
    
    return {'value': '', 'confidence': 0.0}

def extract_provider_info(text: str) -> Dict[str, Any]:
    """
    Extract provider information from text.
    
    Args:
        text: Text to extract provider information from
        
    Returns:
        Dictionary with provider information
    """
    provider_info = {
        'referring_provider': '',
        'interpreting_provider': '',
        'facility': ''
    }
    
    # Extract referring provider
    ref_pattern = r'(?:REFERRING|ORDERED BY|REFERRING PHYSICIAN|REFERRING PROVIDER)[:\s]+([A-Za-z\s.,]+)'
    ref_match = re.search(ref_pattern, text, re.IGNORECASE)
    if ref_match:
        provider_info['referring_provider'] = ref_match.group(1).strip()
    
    # Extract interpreting provider
    int_pattern = r'(?:INTERPRETING|INTERPRETED BY|READING PHYSICIAN|RADIOLOGIST)[:\s]+([A-Za-z\s.,]+)'
    int_match = re.search(int_pattern, text, re.IGNORECASE)
    if int_match:
        provider_info['interpreting_provider'] = int_match.group(1).strip()
    
    # Extract facility
    fac_pattern = r'(?:FACILITY|LOCATION|SITE|PERFORMED AT)[:\s]+([A-Za-z\s.,]+)'
    fac_match = re.search(fac_pattern, text, re.IGNORECASE)
    if fac_match:
        provider_info['facility'] = fac_match.group(1).strip()
    
    return provider_info

def extract_signed_by(text: str) -> str:
    """
    Extract signature information from text.
    
    Args:
        text: Text to extract signature from
        
    Returns:
        Signature text or empty string if not found
    """
    signature_patterns = [
        r'(?:ELECTRONICALLY SIGNED BY|SIGNED BY|SIGNATURE)[:\s]+([A-Za-z\s.,]+)',
        r'(?:DICTATED BY|REPORTED BY)[:\s]+([A-Za-z\s.,]+)'
    ]
    
    for pattern in signature_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    return ""

def process_document_text(text: str) -> Dict[str, Any]:
    """
    Process document text to extract structured data.
    
    Args:
        text: Document text to process
        
    Returns:
        Dictionary with extracted structured data
    """
    structured_data = {
        'patient_name': "Not Available",
        'exam_date': "Not Available",
        'exam_type': "Not Available",
        'birads_score': "Not Available",
        'findings': "",
        'impression': "",
        'recommendation': "",
        'clinical_history': "",
        'provider_info': {},
        'signed_by': ""
    }
    
    # Extract patient info
    patient_info = extract_patient_info(text)
    if patient_info.get('name'):
        structured_data['patient_name'] = patient_info['name']
    
    # Extract exam date
    exam_date = extract_date_from_text(text)
    if exam_date:
        structured_data['exam_date'] = exam_date
    
    # Extract exam type
    exam_type = extract_exam_type(text)
    if exam_type:
        structured_data['exam_type'] = exam_type
    
    # Extract BIRADS score
    birads_info = extract_birads_score(text)
    if birads_info['value']:
        structured_data['birads_score'] = birads_info['value']
    
    # Extract sections
    sections = extract_sections(text)
    for key in ['findings', 'impression', 'recommendation', 'clinical_history']:
        if key in sections:
            structured_data[key] = sections[key]
    
    # Extract provider info
    structured_data['provider_info'] = extract_provider_info(text)
    
    # Extract signature
    structured_data['signed_by'] = extract_signed_by(text)
    
    return structured_data

def extract_structured_data(text: str) -> Dict[str, Any]:
    """
    Alias for process_document_text for backward compatibility.
    
    Args:
        text: Document text to process
        
    Returns:
        Dictionary with extracted structured data
    """
    return process_document_text(text)
