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
    'extract_section',
    'detect_report_language',
    'clean_provider_name'
]

class BiradsSpellingCorrector:
    """
    Class for correcting common misspellings of BIRADS and related terminology.
    
    This class provides methods to detect and correct misspellings of BIRADS
    terminology in medical texts, improving detection of BIRADS scores in OCR output.
    Now supports both English and French medical terminology.
    """
    
    def __init__(self):
        # Dictionary of common misspellings and their corrections
        self.en_corrections = {
            # BIRADS variations
            'binds': 'birads',
            'bi-rads': 'birads',
            'birods': 'birads',
            'bi rads': 'birads',
            'birad': 'birads',
            'birads': 'birads',
            'brads': 'birads',
            'btrads': 'birads',
            'bi-rad': 'birads',
            'braid': 'birads',
            'baird': 'birads',
            'blrads': 'birads',
            'bil-rads': 'birads',
            'bird': 'birads',
            'birdos': 'birads',
            'bi-rad$': 'birads',
            'brirads': 'birads',
            'birats': 'birads',
            'bi rad': 'birads',
            'bira ds': 'birads',
            'bi-r ads': 'birads',
            'b irads': 'birads',
            'br-irads': 'birads',
            
            # ACR variations
            'acr': 'acr',
            'acn': 'acr',
            'aci': 'acr',
            'ocr': 'acr',
            'acf': 'acr',
            
            # Category variations
            'category': 'category',
            'cat': 'category',
            'cot': 'category',
            'cotegory': 'category',
            'catergoy': 'category',
            'catagory': 'category',
            'categorv': 'category',
        }
        
        # French corrections for BIRADS terminology
        self.fr_corrections = {
            # French BIRADS variations 
            'birads': 'birads',
            'bi-rads': 'birads',
            'bi rads': 'birads',
            'birad': 'birads',
            'blrads': 'birads',
            'bil-rads': 'birads',
            'bi rad': 'birads',
            'b-irads': 'birads',
            
            # French category variations
            'catégorie': 'category',
            'categorie': 'category',
            'classification': 'category',
            'classe': 'category',
            'cat': 'category',
            
            # French ACR variations
            'acr': 'acr',
            'a.c.r': 'acr',
            'a-c-r': 'acr',
            'a c r': 'acr',
            
            # French context words
            'sein': 'breast',
            'mammographie': 'mammogram',
            'échographie': 'ultrasound',
            'résultat': 'result',
            'conclusion': 'impression',
            'interprétation': 'impression'
        }
        
        # Calculate edit distance threshold based on term length
        self.threshold_func = lambda word: 1 if len(word) <= 4 else (2 if len(word) <= 8 else 3)
    
    def correct(self, word: str) -> str:
        """
        Correct a potentially misspelled BIRADS-related term.
        
        Args:
            word: Word to correct
            
        Returns:
            Corrected word or original if no correction found
        """
        if not word:
            return word
            
        # Lowercase for comparison
        word_lower = word.lower()
        
        # Check if it's an exact match in English corrections
        if word_lower in self.en_corrections:
            return self.en_corrections[word_lower]
            
        # Check if it's an exact match in French corrections
        if word_lower in self.fr_corrections:
            return self.fr_corrections[word_lower]
        
        # If not exact match, try to find closest match using edit distance
        # First try English corrections
        for correct_word, correction in self.en_corrections.items():
            threshold = self.threshold_func(correct_word)
            if edit_distance(word_lower, correct_word) <= threshold:
                return correction
                
        # Then try French corrections
        for correct_word, correction in self.fr_corrections.items():
            threshold = self.threshold_func(correct_word)
            if edit_distance(word_lower, correct_word) <= threshold:
                return correction
        
        # No suitable correction found
        return word
    
    def calculate_context_score(self, text: str) -> float:
        """
        Calculate a context score indicating how likely the text contains BIRADS information.
        Higher score means BIRADS information is more likely present.
        
        Args:
            text: Text to analyze
            
        Returns:
            Context score between 0.0 and 1.0
        """
        if not text:
            return 0.0
            
        # Keywords that suggest BIRADS context is present
        # English keywords
        en_keywords = [
            'mammogram', 'breast', 'ultrasound', 'imaging', 'radiologist',
            'finding', 'lesion', 'mass', 'calcification', 'density',
            'assessment', 'follow-up', 'biopsy', 'screening', 'diagnostic'
        ]
        
        # French keywords
        fr_keywords = [
            'mammographie', 'sein', 'échographie', 'imagerie', 'radiologue',
            'résultat', 'lésion', 'masse', 'calcification', 'densité',
            'évaluation', 'suivi', 'biopsie', 'dépistage', 'diagnostic'
        ]
        
        # Count keyword occurrences
        en_count = sum(1 for kw in en_keywords if re.search(r'\b' + kw + r'\b', text, re.IGNORECASE))
        fr_count = sum(1 for kw in fr_keywords if re.search(r'\b' + kw + r'\b', text, re.IGNORECASE))
        
        # Use the higher count (English or French)
        keyword_count = max(en_count, fr_count)
        
        # Basic normalization - score between 0 and 1
        context_score = min(1.0, keyword_count / 10.0)
        
        # Additional boost for specific strong indicators
        if re.search(r'\b(?:(?:bi-?)?rads|acr)(?:\s+|:).{0,10}(?:[0-6][abc]?)\b', text, re.IGNORECASE):
            context_score += 0.3
            
        # Cap at 1.0
        return min(1.0, context_score)
    
    def find_and_correct_birads_term(self, text: str) -> tuple:
        """
        Finds and corrects BIRADS terms in the text.
        
        Args:
            text: Text to process
            
        Returns:
            Tuple of (corrected text, list of corrections made)
        """
        if not text:
            return text, []
            
        # Get language of the text
        lang = detect_report_language(text)
        
        # Detect both English and French BIRADS-related terms
        # For BIRADS/ACR/category terms 
        birads_pattern = r'\b(bi[ -]?r?a?ds?|blrads|brrads|birods|brads|braids|bi[ -]?rad)\b'
        acr_pattern = r'\b(acr|a\.c\.r|a-c-r)\b'
        category_pattern = r'\b(cat(?:egory|egorie)?|class(?:ification|e)?|catégorie)\b'
        
        corrections = []
        
        # Helper function to make replacements
        def replace_match(match):
            original = match.group(0)
            term_type = match.lastgroup
            
            # Get appropriate dictionary based on language
            corrections_dict = self.fr_corrections if lang == 'fr' else self.en_corrections
            
            if term_type == 'birads':
                corrected = 'BIRADS'
            elif term_type == 'acr':
                corrected = 'ACR'
            elif term_type == 'category':
                corrected = 'Category' if lang == 'en' else 'Catégorie'
            else:
                # Use appropriate correction from dictionary
                corrected = corrections_dict.get(original.lower(), original)
                
            if corrected.lower() != original.lower():
                corrections.append((original, corrected))
                
            return corrected
        
        # Combined pattern with named groups
        combined_pattern = f'(?P<birads>{birads_pattern})|(?P<acr>{acr_pattern})|(?P<category>{category_pattern})'
        
        # Make replacements
        corrected_text = re.sub(combined_pattern, replace_match, text, flags=re.IGNORECASE)
        
        return corrected_text, corrections

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
    corrected_text, corrections = corrector.find_and_correct_birads_term(text)
    
    if corrections:
        logging.info(f"Corrected BIRADS terminology with {len(corrections)} corrections")
    
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

def detect_report_language(text: str) -> str:
    """
    Detect if a medical report is in French or English based on medical terminology.
    
    Args:
        text: The report text
        
    Returns:
        'fr' for French, 'en' for English
    """
    if not text:
        return "en"
        
    # Key French medical terms with accents
    fr_terms = [
        r'résultat', r'mammographie', r'échographie', r'sein', r'médecin', 
        r'recommandation', r'suivi', r'catégorie', r'interprétation', 
        r'clinique', r'antécédent', r'examen'
    ]
    
    # Key English medical terms
    en_terms = [
        r'finding', r'mammogram', r'ultrasound', r'breast', r'physician',
        r'recommendation', r'follow-up', r'category', r'interpretation',
        r'clinical', r'history', r'examination'
    ]
    
    # Count term occurrences
    fr_count = sum(len(re.findall(term, text, re.IGNORECASE)) for term in fr_terms)
    en_count = sum(len(re.findall(term, text, re.IGNORECASE)) for term in en_terms)
    
    # Add section header checks
    fr_headers = sum(len(re.findall(h, text, re.IGNORECASE)) for h in 
                    [r'RÉSULTATS', r'CONCLUSION', r'RECOMMANDATION'])
    en_headers = sum(len(re.findall(h, text, re.IGNORECASE)) for h in 
                    [r'FINDINGS', r'IMPRESSION', r'RECOMMENDATION'])
    
    fr_count += fr_headers * 2  # Weight headers more heavily
    en_count += en_headers * 2
    
    # Return language with higher score, default to English
    return 'fr' if fr_count > en_count else 'en'

def extract_provider_info(text: str) -> Dict[str, Any]:
    """
    Extract provider information from text with enhanced bilingual support.
    
    This function extracts referring provider, interpreting provider, and facility
    with support for both English and French medical reports.
    
    Args:
        text: Text to extract provider information from
        
    Returns:
        Dictionary with provider information
    """
    if text is None:
        raise ValueError("Input text cannot be None")
        
    if not text.strip():
        logging.debug("Empty text provided to extract_provider_info")
        return {"referring_provider": {"value": "", "confidence": 0.0},
                "interpreting_provider": {"value": "", "confidence": 0.0},
                "facility": {"value": "", "confidence": 0.0}}
    
    # Detect language
    lang = detect_report_language(text)
    
    # Define bilingual patterns
    referring_patterns = {
        'en': [
            (r'(?:REFERRING|ORDERED BY|REFERRING PHYSICIAN|REFERRING PROVIDER)[:\s]+([A-Za-z\s.,\-]+(?:MD|M\.D\.|DO|PhD)?)', 0.9),
            (r'(?:ORDERING PHYSICIAN|PHYSICIAN OF RECORD)[:\s]+([A-Za-z\s.,\-]+(?:MD|M\.D\.|DO|PhD)?)', 0.85)
        ],
        'fr': [
            (r'(?:MÉDECIN RÉFÉRENT|PRESCRIPTEUR|MÉDECIN TRAITANT|RÉFÉRÉ PAR)[:\s]+([A-Za-zÀ-ÿ\s.,\-]+(?:MD|M\.D\.|DO|PhD)?)', 0.9),
            (r'(?:DEMANDÉ PAR|MÉDECIN DEMANDEUR)[:\s]+([A-Za-zÀ-ÿ\s.,\-]+(?:MD|M\.D\.|DO|PhD)?)', 0.85)
        ]
    }
    
    interpreting_patterns = {
        'en': [
            (r'(?:INTERPRETING|INTERPRETED BY|READING PHYSICIAN|RADIOLOGIST)[:\s]+([A-Za-z\s.,\-]+(?:MD|M\.D\.|DO|PhD)?)', 0.9),
            (r'(?:SIGNED BY|ELECTRONICALLY SIGNED BY)[:\s]+([A-Za-z\s.,\-]+(?:MD|M\.D\.|DO|PhD)?)', 0.9)
        ],
        'fr': [
            (r'(?:INTERPRÉTÉ PAR|RADIOLOGUE|MÉDECIN INTERPRÉTANT)[:\s]+([A-Za-zÀ-ÿ\s.,\-]+(?:MD|M\.D\.|DO|PhD)?)', 0.9),
            (r'(?:SIGNÉ PAR|SIGNATURE ÉLECTRONIQUE)[:\s]+([A-Za-zÀ-ÿ\s.,\-]+(?:MD|M\.D\.|DO|PhD)?)', 0.9)
        ]
    }
    
    facility_patterns = {
        'en': [
            (r'(?:FACILITY|LOCATION|SITE|PERFORMED AT)[:\s]+([^\n]+)', 0.8)
        ],
        'fr': [
            (r'(?:ÉTABLISSEMENT|LIEU|SITE|CLINIQUE|HÔPITAL)[:\s]+([^\n]+)', 0.8)
        ]
    }
    
    results = {
        "referring_provider": {"value": "", "confidence": 0.0},
        "interpreting_provider": {"value": "", "confidence": 0.0},
        "facility": {"value": "", "confidence": 0.0}
    }
    
    # Process each pattern type
    pattern_mappings = [
        ("referring_provider", referring_patterns),
        ("interpreting_provider", interpreting_patterns),
        ("facility", facility_patterns)
    ]
    
    for field, patterns_dict in pattern_mappings:
        # Try primary language first
        for pattern, confidence in patterns_dict.get(lang, []):
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value = clean_provider_name(match.group(1).strip(), lang)
                results[field] = {"value": value, "confidence": confidence}
                break
                
        # If not found, try secondary language
        if not results[field]["value"]:
            secondary_lang = 'en' if lang == 'fr' else 'fr'
            for pattern, confidence in patterns_dict.get(secondary_lang, []):
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    value = clean_provider_name(match.group(1).strip(), secondary_lang)
                    results[field] = {"value": value, "confidence": confidence * 0.9}  # Slightly lower confidence for secondary language
                    break
    
    # Fall back to extract_signed_by for interpreting provider if not found
    if not results["interpreting_provider"]["value"]:
        signed_by = extract_signed_by(text)
        if signed_by:
            results["interpreting_provider"] = {"value": signed_by, "confidence": 0.7}
    
    return results

def clean_provider_name(provider_name: str, lang: str) -> str:
    """
    Clean provider name by handling titles and formatting properly.
    Addresses French hyphenated names and titles.
    
    Args:
        provider_name: Raw provider name
        lang: Language code ('en' or 'fr')
        
    Returns:
        Cleaned provider name
    """
    # Remove common titles
    titles = {
        'en': [r'Dr\.?\s+', r'Professor\s+', r'MD\s+', r'M\.D\.\s+', r'PhD\s+'],
        'fr': [r'Dr\.?\s+', r'Pr\.?\s+', r'Professeur\s+', r'MD\s+', r'M\.D\.\s+']
    }
    
    name = provider_name
    for title in titles[lang] + titles['en' if lang == 'fr' else 'fr']:
        name = re.sub(title, '', name, flags=re.IGNORECASE)
    
    # Handle French hyphenated names - preserve hyphens in names like Jean-Marie
    if lang == 'fr':
        # Make sure hyphens between names are preserved
        name = re.sub(r'(\w)\s*-\s*(\w)', r'\1-\2', name)
    
    # Remove trailing titles or credentials
    name = re.sub(r',?\s*(?:MD|M\.D\.|DO|PhD|FRCPC)$', '', name, flags=re.IGNORECASE)
    
    return name.strip()

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
    Extract a section from text with bilingual support and dynamic end detection.
    
    Args:
        text: Text to extract section from
        section_name: Name of the section to extract (e.g., 'findings', 'impression')
        
    Returns:
        Extracted section text or empty string if not found
    """
    if not text:
        return ""
    
    # Detect language
    lang = detect_report_language(text)
    
    # Define bilingual section markers with variations and colons
    bilingual_section_markers = {
        'findings': {
            'en': [r'(?:FINDINGS|FINDING|OBSERVATIONS?|RESULTS?|REPORT)(?:\s*:|\s*\n)'],
            'fr': [r'(?:RÉSULTATS|CONSTATATIONS|OBSERVATIONS|DONNÉES|EXAMEN)(?:\s*:|\s*\n)'] 
        },
        'impression': {
            'en': [r'(?:IMPRESSIONS?|CONCLUSIONS?|INTERPRETATION|ASSESSMENT|SUMMARY)(?:\s*:|\s*\n)'],
            'fr': [r'(?:IMPRESSION|CONCLUSION|INTERPRÉTATION|ÉVALUATION|SYNTHÈSE)(?:\s*:|\s*\n)']
        },
        'recommendation': {
            'en': [r'(?:RECOMMENDATIONS?|FOLLOW-?UP|ADVICE|PLAN)(?:\s*:|\s*\n)'],
            'fr': [r'(?:RECOMMANDATIONS?|SUIVI|CONSEIL|CONDUITE À TENIR|PLAN)(?:\s*:|\s*\n)']
        },
        'clinical_history': {
            'en': [r'(?:CLINICAL\s+HISTORY|PATIENT\s+HISTORY|HISTORY|INDICATION|CLINICAL\s+INDICATION)(?:\s*:|\s*\n)'],
            'fr': [r'(?:ANTÉCÉDENTS|HISTOIRE\s+CLINIQUE|ANAMNÈSE|INDICATION|RENSEIGNEMENTS\s+CLINIQUES)(?:\s*:|\s*\n)']
        },
        'patient_history': {
            'en': [r'(?:PATIENT\s+HISTORY|HISTORY\s+OF\s+PRESENT\s+ILLNESS|CLINICAL\s+INFORMATION)(?:\s*:|\s*\n)'],
            'fr': [r'(?:ANTÉCÉDENTS\s+DU\s+PATIENT|HISTOIRE\s+DE\s+LA\s+MALADIE|INFORMATION\s+CLINIQUE)(?:\s*:|\s*\n)']
        }
    }
    
    # Get primary and fallback markers for the requested section
    primary_markers = bilingual_section_markers.get(section_name.lower(), {}).get(lang, [])
    fallback_markers = bilingual_section_markers.get(section_name.lower(), {}).get('en' if lang == 'fr' else 'fr', [])
    
    # Combine markers with primary language first
    all_markers = primary_markers + fallback_markers
    
    # Try each marker pattern
    for marker in all_markers:
        match = re.search(marker, text, re.IGNORECASE)
        if not match:
            continue
            
        # Find the start position (end of the marker)
        start_pos = match.end()
        
        # Define section patterns based on language
        if lang == 'fr':
            next_section_pattern = r'\n(?:[A-ZÀÁÂÄÆÇÈÉÊËÌÍÎÏÒÓÔÖÙÚÛÜÝŸ][A-ZÀÁÂÄÆÇÈÉÊËÌÍÎÏÒÓÔÖÙÚÛÜÝŸa-zàáâäæçèéêëìíîïòóôöùúûüýÿ\s]{3,}:|\n\s*[A-ZÀÁÂÄÆÇÈÉÊËÌÍÎÏÒÓÔÖÙÚÛÜÝŸ][A-ZÀÁÂÄÆÇÈÉÊËÌÍÎÏÒÓÔÖÙÚÛÜÝŸa-zàáâäæçèéêëìíîïòóôöùúûüýÿ\s]{3,}:)'
        else:
            next_section_pattern = r'\n(?:[A-Z][A-Z\s]{3,}:|\n\s*[A-Z][A-Z\s]{3,}:)'
        
        # Find the next section header
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
