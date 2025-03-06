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
    'extract_structured_data',
    'extract_section'
]

class BiradsSpellingCorrector:
    """
    Specialized spelling corrector for BIRADS and related medical terms.
    
    This class provides targeted correction for common OCR errors and typos
    in BIRADS terminology in medical reports.
    """
    
    def __init__(self):
        # Dictionary of common misspellings and their corrections
        self.corrections = {
            # BIRADS variations
            'blrads': 'birads',
            'bi-rads': 'birads',
            'bi rads': 'birads',
            'birad': 'birads',
            'birads': 'birads',  # correct spelling as reference
            'bi-rad': 'birads',
            'brads': 'birads',
            'bl-rads': 'birads',
            'bl rads': 'birads',
            'b1-rads': 'birads',
            'bl-rad': 'birads',
            'bidrads': 'birads',
            'bil-rads': 'birads',
            'bir-ads': 'birads',
            'bi-ras': 'birads',
            'bl-ras': 'birads',
            
            # ACR variations
            'acr': 'acr',  # correct spelling as reference
            'arc': 'acr',
            'aor': 'acr',
            'acn': 'acr',
            
            # Category variations
            'categor': 'category',
            'catégorie': 'category',
            'categorie': 'category',
            'catagory': 'category',
            'categ': 'category',
            'cat': 'category',
            
            # Classification variations
            'classif': 'classification',
            'clasif': 'classification',
            'class': 'classification',
            'classe': 'classification',
        }
        
        # Dictionary to track common co-occurrence terms that help identify BIRADS context
        self.context_terms = {
            'mammogram': 2.0,
            'mammography': 2.0,
            'breast': 1.5,
            'ultrasound': 1.5,
            'assessment': 1.5,
            'category': 2.0,
            'score': 2.0,
            'classification': 2.0,
            'impression': 1.0,
            'finding': 1.0,
            'suspicious': 1.0,
            'benign': 1.0,
            'malignant': 1.0,
            'biopsy': 1.0,
            'radiologist': 0.5,
            'follow-up': 0.5,
            'screening': 0.5,
        }
    
    def correct(self, word: str) -> str:
        """
        Correct known misspellings of BIRADS and related terms.
        
        Args:
            word: Word to correct
            
        Returns:
            Corrected version of the word, or original if no correction needed
        """
        word_lower = word.lower()
        if word_lower in self.corrections:
            return self.corrections[word_lower]
        return word
    
    def calculate_context_score(self, text: str) -> float:
        """
        Calculate context score for likelihood that text contains BIRADS information.
        
        Args:
            text: Text to analyze
        
    Returns:
            Score indicating confidence that text contains BIRADS information
        """
        text_lower = text.lower()
        score = 0.0
        
        # Look for contextual terms that suggest BIRADS context
        for term, term_weight in self.context_terms.items():
            if term in text_lower:
                score += term_weight
        
        # Look for score patterns (0-6 with optional a, b, c)
        if re.search(r'\b[0-6][abc]?\b', text_lower):
            score += 2.0
            
        # Stronger boost if we find digit patterns near potential BIRADS terms
        # Look for patterns like "... 4 ..." or "... 4a ..." within 10 chars of "rad" etc.
        birads_digit_pattern = re.compile(r'(?:(?:bi|bl|b)(?:[-\s])?r?a?d|acr|cat).{1,10}?([0-6][abc]?)|([0-6][abc]?).{1,10}?(?:(?:bi|bl|b)(?:[-\s])?r?a?d|acr|cat)', re.IGNORECASE)
        if birads_digit_pattern.search(text_lower):
            score += 5.0
            
        return min(score, 10.0)  # Cap at 10.0
    def find_and_correct_birads_term(self, text: str) -> tuple:
        """
        Find potential BIRADS terms in text and correct them.
        
        Args:
            text: Text to analyze
        
        Returns:
            Tuple of (corrected_text, was_corrected, confidence)
        """
        # Skip empty text
        if not text:
            return text, False, 0.0
        
        # Calculate context score first
        context_score = self.calculate_context_score(text) / 10.0  # Normalize to 0-1
        
        # Early return if context score is very low
        if context_score < 0.2:
            return text, False, context_score
        
        # Words that could be misspelled BIRADS terms
        potential_birads_patterns = [
            # Look for common OCR errors for BIRADS
            r'\b(?:BI|BL|B)(?:[-\s])?R?A?DS\b',
            r'\b(?:BI|BL|B)(?:[-\s])?R?A?D\b',
            # ACR patterns
            r'\b(?:ACR|ARC|AOR|ACN)\b',
            # BI-RADS categories directly
            r'\bcate(?:gor)?(?:y|ie)?\s+([0-6][abc]?)\b',
            r'\bclass(?:if|ificat)(?:ion|e)?\s+([0-6][abc]?)\b',
            # Potentially misspelled with numbers
            r'\b(?:81|B1|8I)-?(?:RADS|RAD)\b',
        ]
        
        corrected_text = text
        was_corrected = False
        highest_confidence = context_score  # Start with context score as baseline
        
        for pattern in potential_birads_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                matched_text = match.group(0)
                
                # Tokenize the matched text
                words = re.findall(r'\b[a-zA-Z]+\b', matched_text)
                
                # Correct each word
                for word in words:
                    corrected_word = self.correct(word)
                    if corrected_word != word.lower():
                        # Apply correction, preserving original case pattern if possible
                        if word.isupper():
                            corrected_word = corrected_word.upper()
                        elif word[0].isupper():
                            corrected_word = corrected_word.capitalize()
                            
                        # Replace only this instance (not all occurrences)
                        corrected_text = corrected_text.replace(matched_text, 
                                                              matched_text.replace(word, corrected_word), 1)
                        was_corrected = True
                        
                        # Increase confidence when we make a correction
                        correction_confidence = 0.85  # Base confidence in the correction
                        highest_confidence = max(highest_confidence, correction_confidence)
        
        return corrected_text, was_corrected, highest_confidence

def preprocess_text_for_extraction(text: str) -> str:
    """
    Preprocess text to improve extraction quality.
    
    Args:
        text: Raw OCR text to preprocess
        
    Returns:
        Preprocessed text
    """
    if not text:
        return ""
    
    # Initialize our medical spelling corrector
    corrector = BiradsSpellingCorrector()
    
    # Correct common OCR errors in BIRADS terminology
    corrected_text, was_corrected, confidence = corrector.find_and_correct_birads_term(text)
    
    if was_corrected:
        logging.info(f"Corrected BIRADS terminology with confidence {confidence:.2f}")
    
    # Normalize whitespace
    corrected_text = re.sub(r'\s+', ' ', corrected_text)
    
    # Fix common OCR errors in dates
    # Replace 'l' with '1' in date patterns
    corrected_text = re.sub(r'(\d{1,2})/l/(\d{4})', r'\1/1/\2', corrected_text)
    corrected_text = re.sub(r'(\d{1,2})-l-(\d{4})', r'\1-1-\2', corrected_text)
    
    # Replace 'O' with '0' in date patterns
    corrected_text = re.sub(r'(\d{1,2})/O/(\d{4})', r'\1/0/\2', corrected_text)
    corrected_text = re.sub(r'(\d{1,2})-O-(\d{4})', r'\1-0-\2', corrected_text)
    
    # Fix broken line issues in critical data (BIRADS, dates, etc.)
    # This helps when OCR breaks these across lines
    corrected_text = re.sub(r'(BI-?RA?DS)\s+(\d)', r'\1 \2', corrected_text, flags=re.IGNORECASE)
    corrected_text = re.sub(r'(ACR)\s+(\d)', r'\1 \2', corrected_text, flags=re.IGNORECASE)
    
    return corrected_text

def extract_birads_score(text: str) -> Dict[str, Any]:
    """
    Extract BIRADS score from text using enhanced pattern matching with typo correction.
    
    This function searches for multiple patterns to identify BIRADS scores in various
    formats, including handling common OCR errors and typos. The function first applies
    spelling correction to handle common typos like "BLRADS", then applies a series of
    patterns to extract the score.
    
    Args:
        text: Text to extract BIRADS score from
        
    Returns:
        Dictionary with 'value' and 'confidence' keys, or empty value and 0 confidence
        if no score is found
        
    Raises:
        ValueError: If input text is None
    """
    if text is None:
        raise ValueError("Input text cannot be None")
        
    if not text.strip():
        logging.debug("Empty text provided to extract_birads_score")
        return {"value": "", "confidence": 0.0}
    
    # Apply preprocessing and spelling correction
    processed_text = preprocess_text_for_extraction(text)
    
    # Initialize patterns with confidence scores
    birads_patterns = [
        # Standard BIRADS patterns
        (r'(?:BI-?RADS|BIRADS)(?:\s+|:|\s+CATEGORY\s+|\s+SCORE\s+|[\s:]+)([0-6][a-c]?)', 0.95),
        (r'(?:BI-?RADS|BIRADS)\s+(?:classification|assessment|category|class)(?:\s+|:|\s+is\s+)([0-6][a-c]?)', 0.95),
        (r'(?:BI-?RADS|BIRADS)(?:\s*[:#=]\s*)([0-6][a-c]?)', 0.95),
        
        # ACR equivalents
        (r'ACR(?:\s+|:|\s+CATEGORY\s+|\s+SCORE\s+|[\s:]+)([0-6][a-c]?)', 0.9),
        (r'ACR\s+(?:classification|assessment|category|class)(?:\s+|:|\s+is\s+)([0-6][a-c]?)', 0.9),
        (r'ACR(?:\s*[:#=]\s*)([0-6][a-c]?)', 0.9),
        
        # Category formats
        (r'CATEGORY\s*(?:is|:|=|\s)\s*([0-6][a-c]?)', 0.85),
        (r'CATEGORY\s+(?:BI-?RADS|BIRADS|ACR)\s*[:#=]?\s*([0-6][a-c]?)', 0.9),
        (r'(?:BI-?RADS|BIRADS|ACR)\s+CATEGORY\s*[:#=]?\s*([0-6][a-c]?)', 0.9),
        
        # Assessment formats
        (r'ASSESSMENT\s*(?:CATEGORY)?\s*[:#=]?\s*(?:BI-?RADS|BIRADS|ACR)?\s*[:#=]?\s*([0-6][a-c]?)', 0.85),
        (r'ASSESSMENT\s*[:#=]?\s*(?:BI-?RADS|BIRADS|ACR)?\s*CATEGORY\s*[:#=]?\s*([0-6][a-c]?)', 0.85),
        
        # Mixed case versions (slightly lower confidence)
        (r'(?i)(?:BI-?RADS|BIRADS)(?:\s+|:|\s+category\s+|\s+score\s+|[\s:]+)([0-6][a-c]?)', 0.85),
        (r'(?i)ACR(?:\s+|:|\s+category\s+|\s+score\s+|[\s:]+)([0-6][a-c]?)', 0.8),
        (r'(?i)category\s*(?:is|:|=|\s)\s*([0-6][a-c]?)', 0.75),
        
        # Secondary patterns - contextual analysis
        (r'interpreted\s+as\s+(?:BI-?RADS|BIRADS|ACR)(?:\s+|:)([0-6][a-c]?)', 0.85),
        (r'(?:finding|impression|assessment)(?:\s+is|\s+are|\s+shows|\s+demonstrates|\s+represents)(?:[^.]*?)(?:BI-?RADS|BIRADS|ACR)(?:\s+|:)([0-6][a-c]?)', 0.8),
    ]
    
    # First pass: try to find exact matches using our patterns
    for pattern, confidence in birads_patterns:
        for match in re.finditer(pattern, processed_text, re.IGNORECASE):
            birads_value = match.group(1).strip()
            # Return just the numeric value instead of "BIRADS X"
            logging.debug(f"Extracted BIRADS score '{birads_value}' with confidence {confidence}")
            return {"value": birads_value, "confidence": confidence}
    
    # If direct extraction failed, try secondary patterns that look for BI-RADS scores in context
    secondary_patterns = [
        # Secondary patterns for inferring BIRADS scores from content
        (r'(?:findings|appearance)(?:[^.]*?)consistent\s+with\s+(?:BI-?RADS|BIRADS|ACR)(?:\s+|:)([0-6][a-c]?)', 0.8),
        (r'recommend(?:ed|ing)?\s+(?:BI-?RADS|BIRADS|ACR)(?:\s+|:)([0-6][a-c]?)', 0.8),
        (r'(?:BI-?RADS|BIRADS|ACR)(?:[^.]*?)(?:recommended|indicated|assigned|designated)(?:[^.]*?)([0-6][a-c]?)', 0.75),
        
        # Last resort: try to find isolated BIRADS scores
        (r'(?<!\w)(?:BI-?RADS|BIRADS|ACR)\s+([0-6][a-c]?)(?!\w)', 0.7),
        (r'(?<!\w)(?:BI-?RADS|BIRADS|ACR)(?:\s+|\s*[:#=]\s*)([0-6][a-c]?)(?!\w)', 0.7),
    ]
    
    for pattern, confidence in secondary_patterns:
        for match in re.finditer(pattern, processed_text, re.IGNORECASE):
            birads_value = match.group(1).strip()
            # Return just the numeric value instead of "BIRADS X"
            logging.debug(f"Extracted BIRADS score '{birads_value}' from secondary pattern with confidence {confidence}")
            return {"value": birads_value, "confidence": confidence}
    
    # Third pass: try to infer BIRADS from clinical statements if no direct score is found
    if re.search(r'normal\s+mammogram|no\s+evidence\s+of\s+malignancy|negative\s+(?:for|finding|examination)', 
                processed_text, re.IGNORECASE):
        return {"value": "1", "confidence": 0.7}
    
    if re.search(r'benign\s+finding|benign\s+appearance|typical\s+(?:benign|appearance)|stable', 
                processed_text, re.IGNORECASE):
        return {"value": "2", "confidence": 0.7}
    
    if re.search(r'probably\s+benign|likely\s+benign|short(?:\s|-)?term\s+follow(?:\s|-)?up', 
                processed_text, re.IGNORECASE):
        return {"value": "3", "confidence": 0.7}
    
    if re.search(r'suspicious|biopsy\s+(?:is\s+)?recommended|requires\s+biopsy', 
                processed_text, re.IGNORECASE):
        # Check for 4a/4b/4c patterns
        if re.search(r'low\s+suspicion|mildly\s+suspicious', processed_text, re.IGNORECASE):
            return {"value": "4A", "confidence": 0.65}
        elif re.search(r'moderate(?:ly)?\s+suspicious', processed_text, re.IGNORECASE):
            return {"value": "4B", "confidence": 0.65}
        elif re.search(r'high(?:ly)?\s+suspicious', processed_text, re.IGNORECASE):
            return {"value": "4C", "confidence": 0.65}
        else:
            return {"value": "4", "confidence": 0.7}
    
    if re.search(r'highly\s+suspicious|highly\s+suggestive\s+of\s+malignancy', 
                processed_text, re.IGNORECASE):
        return {"value": "5", "confidence": 0.7}
    
    if re.search(r'biopsy\s+proven\s+malignancy|known\s+malignancy|established\s+cancer', 
                processed_text, re.IGNORECASE):
        return {"value": "6", "confidence": 0.7}
    
    # No BIRADS score found
    logging.debug("No BIRADS score found in text")
    return {"value": "", "confidence": 0.0}

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

def validate_field_types(data):
    """
    Validates and standardizes field types in extracted data.
    
    Args:
        data (dict): Dictionary of extracted data from OCR processing
        
    Returns:
        dict: Dictionary with validated and standardized field types
    """
    if not isinstance(data, dict):
        return data
        
    # Ensure all values are strings or properly formatted
    for key, value in data.items():
        # Skip None values
        if value is None:
            data[key] = "Not Available"
            continue
            
        # Handle dictionary values (likely JSON fields)
        if isinstance(value, dict):
            continue
            
        # Convert all other values to strings
        if not isinstance(value, str):
            try:
                data[key] = str(value)
            except:
                data[key] = "Not Available"
                
    return data

# Add a helper function for section extraction with dynamic end detection
def extract_section(text: str, section_name: str) -> str:
    """
    Extract a section from text with dynamic end detection.
    
    Args:
        text: Text to extract section from
        section_name: Name of the section to extract (e.g., 'findings', 'impression')
        
    Returns:
        Extracted section text or empty string if not found
    """
    if not text:
        return ""
        
    # Define section markers with variations
    section_markers = {
        'findings': [r'(?:FINDINGS|FINDING|OBSERVATIONS?|RESULTS?|REPORT)(?:\s*:|\s*\n)'],
        'impression': [r'(?:IMPRESSIONS?|CONCLUSIONS?|INTERPRETATION|ASSESSMENT|SUMMARY)(?:\s*:|\s*\n)'],
        'recommendation': [r'(?:RECOMMENDATIONS?|FOLLOW-?UP|ADVICE|PLAN)(?:\s*:|\s*\n)'],
        'clinical_history': [r'(?:CLINICAL\s+HISTORY|PATIENT\s+HISTORY|HISTORY|INDICATION|CLINICAL\s+INDICATION)(?:\s*:|\s*\n)'],
        'patient_history': [r'(?:PATIENT\s+HISTORY|HISTORY\s+OF\s+PRESENT\s+ILLNESS|CLINICAL\s+INFORMATION)(?:\s*:|\s*\n)']
    }
    
    # Get markers for the requested section
    markers = section_markers.get(section_name.lower(), [])
    if not markers:
        return ""
    
    # Try each marker pattern
    for marker in markers:
        match = re.search(marker, text, re.IGNORECASE)
        if not match:
            continue
            
        # Find the start position (end of the marker)
        start_pos = match.end()
        
        # Find the next section header (if any)
        next_section_pattern = r'\n(?:[A-Z][A-Z\s]{3,}:|\n\s*[A-Z][A-Z\s]{3,}:)'
        next_match = re.search(next_section_pattern, text[start_pos:])
        
        # Extract the section content
        if next_match:
            end_pos = start_pos + next_match.start()
            section_text = text[start_pos:end_pos].strip()
        else:
            section_text = text[start_pos:].strip()
            
        return section_text
    
    # Section not found
    return ""
