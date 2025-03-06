"""
Text analysis module for Medical Report Processor application.
This module provides functions for analyzing and extracting information from medical text.
"""

import re
import logging
from typing import Dict, Any, Optional, List, Tuple

# Setup logger
logger = logging.getLogger(__name__)

# Define custom exceptions for better error handling
class ExtractionError(Exception):
    """Base exception for extraction errors"""
    pass

class PatternMatchError(ExtractionError):
    """Exception raised when pattern matching fails"""
    pass

class InvalidInputError(ExtractionError):
    """Exception raised when input is invalid"""
    pass

class ProcessingError(ExtractionError):
    """Exception raised when processing fails"""
    pass

# Import functions from pdf_processor.py
try:
    from .pdf_processor import (
        extract_sections,
        extract_patient_info,
        extract_date_from_text,
        extract_exam_type
    )
except ImportError as e:
    logger.error(f"Failed to import functions from pdf_processor: {str(e)}")
    # Define placeholder functions if imports fail
    def extract_sections(text): return {}
    def extract_patient_info(text): return {}
    def extract_date_from_text(text): return ""
    def extract_exam_type(text): return None

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
    
    Raises:
        InvalidInputError: If text is None
        PatternMatchError: If pattern matching fails unexpectedly
    """
    try:
        if text is None:
            logger.warning("extract_birads_score received None input")
            raise InvalidInputError("Input text cannot be None")
            
        if not text:
            logger.debug("extract_birads_score received empty text")
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
            try:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    result = {
                        'value': f'BIRADS {match.group(1)}',
                        'confidence': confidence
                    }
                    logger.debug(f"Extracted BIRADS score: {result['value']} with confidence {confidence}")
                    return result
            except re.error as e:
                logger.error(f"Regex error with pattern '{pattern}': {str(e)}")
                continue
            except Exception as e:
                logger.error(f"Error during pattern matching: {str(e)}")
                continue
        
        logger.debug("No BIRADS score found in text")
        return {'value': '', 'confidence': 0.0}
    
    except InvalidInputError:
        # Re-raise InvalidInputError for specific handling
        raise
    except Exception as e:
        logger.error(f"Unexpected error in extract_birads_score: {str(e)}")
        return {'value': '', 'confidence': 0.0}

def extract_provider_info(text: str) -> Dict[str, Any]:
    """
    Extract provider information from text.
    
    Args:
        text: Text to extract provider information from
        
    Returns:
        Dictionary with provider information
    
    Raises:
        InvalidInputError: If text is None
    """
    provider_info = {
        'referring_provider': '',
        'interpreting_provider': '',
        'facility': ''
    }
    
    try:
        if text is None:
            logger.warning("extract_provider_info received None input")
            raise InvalidInputError("Input text cannot be None")
            
        if not text:
            logger.debug("extract_provider_info received empty text")
            return provider_info
        
        # Extract referring provider
        ref_pattern = r'(?:REFERRING|ORDERED BY|REFERRING PHYSICIAN|REFERRING PROVIDER)[:\s]+([^\n]+)'
        try:
            ref_match = re.search(ref_pattern, text, re.IGNORECASE)
            if ref_match:
                provider_info['referring_provider'] = ref_match.group(1).strip()
                logger.debug(f"Extracted referring provider: {provider_info['referring_provider']}")
        except Exception as e:
            logger.error(f"Error extracting referring provider: {str(e)}")
        
        # Extract interpreting provider
        int_pattern = r'(?:INTERPRETING|INTERPRETED BY|READING PHYSICIAN|RADIOLOGIST)[:\s]+([^\n]+)'
        try:
            int_match = re.search(int_pattern, text, re.IGNORECASE)
            if int_match:
                provider_info['interpreting_provider'] = int_match.group(1).strip()
                logger.debug(f"Extracted interpreting provider: {provider_info['interpreting_provider']}")
        except Exception as e:
            logger.error(f"Error extracting interpreting provider: {str(e)}")
        
        # Extract facility
        fac_pattern = r'(?:FACILITY|LOCATION|SITE|PERFORMED AT)[:\s]+([^\n]+)'
        try:
            fac_match = re.search(fac_pattern, text, re.IGNORECASE)
            if fac_match:
                provider_info['facility'] = fac_match.group(1).strip()
                logger.debug(f"Extracted facility: {provider_info['facility']}")
        except Exception as e:
            logger.error(f"Error extracting facility: {str(e)}")
        
        return provider_info
    
    except InvalidInputError:
        # Re-raise InvalidInputError for specific handling
        raise
    except Exception as e:
        logger.error(f"Unexpected error in extract_provider_info: {str(e)}")
        return provider_info

def extract_signed_by(text: str) -> str:
    """
    Extract signature information from text.
    
    Args:
        text: Text to extract signature from
        
    Returns:
        Signature text or empty string if not found
    
    Raises:
        InvalidInputError: If text is None
    """
    try:
        if text is None:
            logger.warning("extract_signed_by received None input")
            raise InvalidInputError("Input text cannot be None")
            
        if not text:
            logger.debug("extract_signed_by received empty text")
            return ""
        
        signature_patterns = [
            r'(?:ELECTRONICALLY SIGNED BY|SIGNED BY|SIGNATURE)[:\s]+([A-Za-z\s.,]+)',
            r'(?:DICTATED BY|REPORTED BY)[:\s]+([A-Za-z\s.,]+)'
        ]
        
        for pattern in signature_patterns:
            try:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    signature = match.group(1).strip()
                    logger.debug(f"Extracted signature: {signature}")
                    return signature
            except Exception as e:
                logger.error(f"Error with signature pattern '{pattern}': {str(e)}")
                continue
        
        logger.debug("No signature found in text")
        return ""
    
    except InvalidInputError:
        # Re-raise InvalidInputError for specific handling
        raise
    except Exception as e:
        logger.error(f"Unexpected error in extract_signed_by: {str(e)}")
        return ""

def process_document_text(text: str) -> Dict[str, Any]:
    """
    Process document text to extract structured data.
    
    Args:
        text: Document text to process
        
    Returns:
        Dictionary with extracted structured data
    
    Raises:
        InvalidInputError: If text is None
        ProcessingError: If processing fails unexpectedly
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
    
    try:
        if text is None:
            logger.warning("process_document_text received None input")
            raise InvalidInputError("Input text cannot be None")
            
        if not text:
            logger.debug("process_document_text received empty text")
            return structured_data
        
        logger.info("Processing document text")
        
        # Extract patient info
        try:
            patient_info = extract_patient_info(text)
            if patient_info.get('name'):
                structured_data['patient_name'] = patient_info['name']
                logger.debug(f"Extracted patient name: {patient_info['name']}")
        except Exception as e:
            logger.error(f"Error extracting patient info: {str(e)}")
        
        # Extract exam date
        try:
            exam_date = extract_date_from_text(text)
            if exam_date:
                structured_data['exam_date'] = exam_date
                logger.debug(f"Extracted exam date: {exam_date}")
        except Exception as e:
            logger.error(f"Error extracting exam date: {str(e)}")
        
        # Extract exam type
        try:
            exam_type = extract_exam_type(text)
            if exam_type:
                structured_data['exam_type'] = exam_type
                logger.debug(f"Extracted exam type: {exam_type}")
        except Exception as e:
            logger.error(f"Error extracting exam type: {str(e)}")
        
        # Extract BIRADS score
        try:
            birads_info = extract_birads_score(text)
            if birads_info['value']:
                structured_data['birads_score'] = birads_info['value']
                logger.debug(f"Extracted BIRADS score: {birads_info['value']}")
        except Exception as e:
            logger.error(f"Error extracting BIRADS score: {str(e)}")
        
        # Extract sections
        try:
            sections = extract_sections(text)
            for key in ['findings', 'impression', 'recommendation', 'clinical_history']:
                if key in sections:
                    structured_data[key] = sections[key]
                    logger.debug(f"Extracted {key}: {sections[key][:50]}...")
        except Exception as e:
            logger.error(f"Error extracting sections: {str(e)}")
        
        # Extract provider info
        try:
            structured_data['provider_info'] = extract_provider_info(text)
            logger.debug(f"Extracted provider info")
        except Exception as e:
            logger.error(f"Error extracting provider info: {str(e)}")
        
        # Extract signature
        try:
            structured_data['signed_by'] = extract_signed_by(text)
            if structured_data['signed_by']:
                logger.debug(f"Extracted signature: {structured_data['signed_by']}")
        except Exception as e:
            logger.error(f"Error extracting signature: {str(e)}")
        
        logger.info("Document text processing complete")
        return structured_data
    
    except InvalidInputError:
        # Re-raise InvalidInputError for specific handling
        raise
    except Exception as e:
        logger.error(f"Unexpected error in process_document_text: {str(e)}")
        raise ProcessingError(f"Failed to process document: {str(e)}")

def extract_structured_data(text: str) -> Dict[str, Any]:
    """
    Alias for process_document_text for backward compatibility.
    
    Args:
        text: Document text to process
        
    Returns:
        Dictionary with extracted structured data
    """
    try:
        logger.debug("extract_structured_data called, redirecting to process_document_text")
        return process_document_text(text)
    except Exception as e:
        logger.error(f"Error in extract_structured_data: {str(e)}")
        # Return default data on error
        return {
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
