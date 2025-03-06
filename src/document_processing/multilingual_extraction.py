"""
Multilingual extraction module for Medical Report Processor application.

This module provides functionality for language detection and language-specific
extraction patterns for medical reports in different languages.
"""

import re
import logging
from typing import Dict, Any, List, Optional, Tuple
import datetime

# Setup logger
logger = logging.getLogger(__name__)

# Try to import language detection libraries
try:
    from langdetect import detect, LangDetectException
    LANGDETECT_AVAILABLE = True
except ImportError:
    logger.warning("langdetect library not available, falling back to rule-based detection")
    LANGDETECT_AVAILABLE = False

class LanguageDetectionError(Exception):
    """Exception raised when language detection fails"""
    pass

def detect_language(text: str) -> str:
    """
    Detect the language of a medical report.
    
    Args:
        text: The text to analyze
        
    Returns:
        Language code ('en', 'fr', etc.)
        
    Raises:
        LanguageDetectionError: If language detection fails
    """
    if not text:
        logger.warning("Empty text provided for language detection")
        return "en"  # Default to English for empty text
        
    # First try rule-based detection for speed and reliability with medical documents
    french_indicators = [
        r'\bexamen\b', 
        r'\bd[\'"]examen\b', 
        r'mammographie', 
        r'\bsein\b', 
        r'\bseins\b',
        r'\bprotocole\b', 
        r'\bd[\'"]imagerie\b', 
        r'\bconstatations\b', 
        r'\bconclusion\b',
        r'\bdictée\b', 
        r'\btranscrit\b', 
        r'\bréférant\b', 
        r'\bdépistage\b'
    ]
    
    french_count = sum(1 for pattern in french_indicators if re.search(pattern, text, re.IGNORECASE))
    
    # If we have strong evidence of French text
    if french_count >= 3:
        logger.info(f"Detected French document based on {french_count} keyword matches")
        return "fr"
        
    # For less clear cases, use language detection library if available
    if LANGDETECT_AVAILABLE:
        try:
            # Take a reasonable sample from the text for faster detection
            sample = text[:5000]
            lang = detect(sample)
            logger.info(f"Language detection identified: {lang}")
            
            # Map detected language to our supported languages
            if lang == "fr":
                return "fr"
            else:
                return "en"  # Default to English for anything else
                
        except LangDetectException as e:
            logger.warning(f"Language detection failed: {str(e)}")
            # Fall back to English on detection failure
            return "en"
    
    # Default to English if library not available and rule-based detection inconclusive
    return "en"

# French extraction patterns for common medical fields
FRENCH_DATE_PATTERNS = [
    (r'Date d[\'"]examen\s*:\s*(\d{4}-\d{2}-\d{2})', 0.95),
    (r'Date d[\'"]examen\s*:\s*(\d{2}-\d{2}-\d{4})', 0.9),
    (r'Date d[\'"]examen\s*:\s*(\d{1,2}/\d{1,2}/\d{4})', 0.9),
    (r'Date d[\'"]examen\s*:\s*(\d{1,2}[.-]\d{1,2}[.-]\d{4})', 0.9),
    (r'examen[^\n:]*?(\d{1,2}[/-]\d{1,2}[/-]\d{4})', 0.8),
    (r'examen[^\n:]*?(\d{4}[/-]\d{1,2}[/-]\d{1,2})', 0.8)
]

FRENCH_EXAM_TYPE_PATTERNS = [
    (r'MAMMOGRAPHIE\s+(?:DIAGNOSTIQUE|DE DÉPISTAGE|BILATÉRALE)', 'MAMMOGRAM', 0.95),
    (r'mammographie\s+(?:diagnostique|de dépistage|bilatérale)', 'MAMMOGRAM', 0.95),
    (r'mammographie', 'MAMMOGRAM', 0.9),
    (r'ÉCHOGRAPHIE\s+(?:MAMMAIRE|DU SEIN|DES SEINS)', 'ULTRASOUND', 0.95),
    (r'échographie\s+(?:mammaire|du sein|des seins)', 'ULTRASOUND', 0.95),
    (r'TOMOSYNTHÈSE', 'TOMOSYNTHESIS', 0.95),
    (r'tomosynthèse', 'TOMOSYNTHESIS', 0.95),
    (r'IRM\s+(?:MAMMAIRE|DU SEIN|DES SEINS)', 'MRI', 0.95),
    (r'irm\s+(?:mammaire|du sein|des seins)', 'MRI', 0.95)
]

FRENCH_BIRADS_PATTERNS = [
    (r'(?:BI-?RADS|BIRADS).{0,10}?(?:catégorie|classe|classification).{0,5}?(\d[abc]?)', 0.95),
    (r'(?:BI-?RADS|BIRADS).{0,5}?:?\s*(\d[abc]?)', 0.9),
    (r'(?:catégorie|classe|classification).{0,10}?(?:BI-?RADS|BIRADS).{0,5}?(\d[abc]?)', 0.9),
    (r'(?:catégorie|classe|classification).{0,5}?:?\s*(\d[abc]?)', 0.85),
    (r'ACR.{0,5}?(\d[abc]?)', 0.85)
]

FRENCH_FINDING_PATTERNS = [
    (r'(?:Constatations|Observations|Findings|Trouvailles)[\s:]+(.+?)(?=\n\n|\n[A-Z]|\Z)', 0.9),
    (r'(?:CONSTATATIONS|OBSERVATIONS|FINDINGS|TROUVAILLES)[\s:]+(.+?)(?=\n\n|\n[A-Z]|\Z)', 0.9)
]

FRENCH_IMPRESSION_PATTERNS = [
    (r'(?:Impression|Conclusion|Avis|Opinion)[\s:]+(.+?)(?=\n\n|\n[A-Z]|\Z)', 0.9),
    (r'(?:IMPRESSION|CONCLUSION|AVIS|OPINION)[\s:]+(.+?)(?=\n\n|\n[A-Z]|\Z)', 0.9)
]

FRENCH_RECOMMENDATION_PATTERNS = [
    (r'(?:Recommandation|Recommendation|Suivi|Follow-up)[\s:]+(.+?)(?=\n\n|\n[A-Z]|\Z)', 0.9),
    (r'(?:RECOMMANDATION|RECOMMENDATION|SUIVI|FOLLOW-UP)[\s:]+(.+?)(?=\n\n|\n[A-Z]|\Z)', 0.9)
]

FRENCH_PROVIDER_PATTERNS = [
    (r'(?:Médecin|Docteur|Référant|Référent|Référé par)[\s:]+([^.\n]+)', 0.9),
    (r'(?:MÉDECIN|DOCTEUR|RÉFÉRANT|RÉFÉRENT|RÉFÉRÉ PAR)[\s:]+([^.\n]+)', 0.9),
    (r'(?:Radiologiste|Radiologue|Interprété par)[\s:]+([^.\n]+)', 0.9),
    (r'(?:RADIOLOGISTE|RADIOLOGUE|INTERPRÉTÉ PAR)[\s:]+([^.\n]+)', 0.9)
]

FRENCH_FACILITY_PATTERNS = [
    (r'(?:Établissement|Clinique|Hôpital|Centre)[\s:]+([^.\n]+)', 0.9),
    (r'(?:ÉTABLISSEMENT|CLINIQUE|HÔPITAL|CENTRE)[\s:]+([^.\n]+)', 0.9),
    (r'(?:Lieu|Site|Endroit|Facility)[\s:]+([^.\n]+)', 0.9),
    (r'(?:LIEU|SITE|ENDROIT|FACILITY)[\s:]+([^.\n]+)', 0.9)
]

FRENCH_HISTORY_PATTERNS = [
    (r'(?:Histoire|Historique|Antécédents|Renseignements cliniques)[\s:]+(.+?)(?=\n\n|\n[A-Z]|\Z)', 0.9),
    (r'(?:HISTOIRE|HISTORIQUE|ANTÉCÉDENTS|RENSEIGNEMENTS CLINIQUES)[\s:]+(.+?)(?=\n\n|\n[A-Z]|\Z)', 0.9),
    (r'(?:Clinical|Clinique|Information|Informations)[\s:]+(.+?)(?=\n\n|\n[A-Z]|\Z)', 0.85),
    (r'(?:CLINICAL|CLINIQUE|INFORMATION|INFORMATIONS)[\s:]+(.+?)(?=\n\n|\n[A-Z]|\Z)', 0.85)
]

def extract_with_patterns(text: str, patterns: List[Tuple[str, float]]) -> Dict[str, Any]:
    """
    Extract information using a list of patterns with confidence scores.
    
    Args:
        text: Text to extract from
        patterns: List of tuples containing (regex pattern, confidence score)
        
    Returns:
        Dictionary with 'value' and 'confidence' keys
    """
    if not text:
        return {"value": "", "confidence": 0.0}
    
    for pattern, confidence in patterns:
        try:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                value = match.group(1).strip()
                if value:
                    logger.debug(f"Matched pattern with confidence {confidence}: {value[:30]}...")
                    return {"value": value, "confidence": confidence}
        except Exception as e:
            logger.warning(f"Error matching pattern {pattern}: {str(e)}")
    
    return {"value": "", "confidence": 0.0}

def extract_french_date(text: str) -> Dict[str, Any]:
    """Extract date from French medical text"""
    return extract_with_patterns(text, FRENCH_DATE_PATTERNS)

def extract_french_birads(text: str) -> Dict[str, Any]:
    """Extract BIRADS score from French medical text"""
    result = extract_with_patterns(text, FRENCH_BIRADS_PATTERNS)
    if result["value"]:
        result["value"] = f"BIRADS {result['value']}"
    return result

def extract_french_findings(text: str) -> Dict[str, Any]:
    """Extract findings from French medical text"""
    return extract_with_patterns(text, FRENCH_FINDING_PATTERNS)

def extract_french_impression(text: str) -> Dict[str, Any]:
    """Extract impression from French medical text"""
    return extract_with_patterns(text, FRENCH_IMPRESSION_PATTERNS)

def extract_french_recommendation(text: str) -> Dict[str, Any]:
    """Extract recommendation from French medical text"""
    return extract_with_patterns(text, FRENCH_RECOMMENDATION_PATTERNS)

def extract_french_provider(text: str) -> Dict[str, Any]:
    """Extract provider information from French medical text"""
    referring_provider = extract_with_patterns(text, FRENCH_PROVIDER_PATTERNS[:2])
    interpreting_provider = extract_with_patterns(text, FRENCH_PROVIDER_PATTERNS[2:])
    
    provider_info = {
        "referring_provider": referring_provider["value"],
        "interpreting_provider": interpreting_provider["value"],
        "confidence": max(referring_provider["confidence"], interpreting_provider["confidence"])
    }
    
    return provider_info

def extract_french_facility(text: str) -> Dict[str, Any]:
    """Extract facility information from French medical text"""
    return extract_with_patterns(text, FRENCH_FACILITY_PATTERNS)

def extract_french_history(text: str) -> Dict[str, Any]:
    """Extract patient history from French medical text"""
    return extract_with_patterns(text, FRENCH_HISTORY_PATTERNS)

# Aggregated extraction function for French documents
def extract_french_fields(text: str) -> Dict[str, Any]:
    """
    Extract all fields from French medical text.
    
    Args:
        text: French medical text
        
    Returns:
        Dictionary of extracted fields with values and confidence scores
    """
    extracted_data = {}
    
    # Extract date
    date_result = extract_french_date(text)
    if date_result["value"]:
        extracted_data["exam_date"] = date_result["value"]
    
    # Extract BIRADS score
    birads_result = extract_french_birads(text)
    if birads_result["value"]:
        extracted_data["birads_score"] = birads_result["value"]
    
    # Extract findings
    findings_result = extract_french_findings(text)
    if findings_result["value"]:
        extracted_data["findings"] = findings_result["value"]
    
    # Extract impression
    impression_result = extract_french_impression(text)
    if impression_result["value"]:
        extracted_data["impression"] = impression_result["value"]
    
    # Extract recommendation
    recommendation_result = extract_french_recommendation(text)
    if recommendation_result["value"]:
        extracted_data["recommendation"] = recommendation_result["value"]
    
    # Extract provider information
    provider_info = extract_french_provider(text)
    if provider_info["referring_provider"] or provider_info["interpreting_provider"]:
        extracted_data["provider_info"] = {
            "referring_provider": provider_info["referring_provider"],
            "interpreting_provider": provider_info["interpreting_provider"]
        }
    
    # Extract facility
    facility_result = extract_french_facility(text)
    if facility_result["value"]:
        if "provider_info" not in extracted_data:
            extracted_data["provider_info"] = {}
        extracted_data["provider_info"]["facility"] = facility_result["value"]
    
    # Extract history
    history_result = extract_french_history(text)
    if history_result["value"]:
        extracted_data["clinical_history"] = history_result["value"]
    
    logger.info(f"Extracted {len(extracted_data)} fields from French text")
    return extracted_data

def extract_with_language_specific_patterns(text: str, language: str) -> Dict[str, Any]:
    """
    Extract information using language-specific patterns.
    
    Args:
        text: Text to extract from
        language: Language code ('en', 'fr', etc.)
        
    Returns:
        Dictionary of extracted fields
    """
    if not text:
        logger.warning("Empty text provided for extraction")
        return {}
    
    if language == "fr":
        logger.info("Extracting fields using French-specific patterns")
        return extract_french_fields(text)
    else:
        # Default to English patterns - in this case we'll just return an empty dict
        # since English extraction is handled by our main extract_structured_data function
        logger.info("Using standard English extraction patterns")
        return {}

def extract_exam_type_french(text: str) -> Tuple[str, float]:
    """
    Extract exam type from French medical text.
    
    Args:
        text: Text to extract exam type from
        
    Returns:
        Tuple of (normalized exam type, confidence score)
    """
    for pattern, exam_type, confidence in FRENCH_EXAM_TYPE_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            logger.debug(f"Extracted French exam type: {exam_type}")
            return exam_type, confidence
    
    return "", 0.0

def extract_section_french(text: str, section_name: str) -> Tuple[str, float]:
    """
    Extract a specific section from French medical text.
    
    Args:
        text: Text to extract section from
        section_name: Name of section to extract (findings, impression, etc.)
        
    Returns:
        Tuple of (extracted section text, confidence score)
    """
    if section_name not in FRENCH_SECTION_PATTERNS:
        logger.warning(f"Unsupported section name: {section_name}")
        return "", 0.0
    
    for pattern in FRENCH_SECTION_PATTERNS[section_name]:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            section_text = match.group(1).strip()
            logger.debug(f"Extracted French {section_name} section: {section_text[:50]}...")
            return section_text, 0.9
    
    # For French reports, try to extract indications/findings from entire report
    if section_name == "findings" or section_name == "impression":
        # Look for patterns like "Constatations" without explicit sections
        findings_pattern = r'(?:il y a|il existe|on note|on observe|confirment la présence de|présente)([^\.]+)'
        matches = re.finditer(findings_pattern, text, re.IGNORECASE)
        
        findings_text = ""
        for match in matches:
            findings_text += match.group(1).strip() + " "
        
        if findings_text:
            return findings_text.strip(), 0.7
    
    return "", 0.0

def extract_history_french(text: str) -> Tuple[str, float]:
    """
    Extract patient history from French medical text.
    
    Args:
        text: Text to extract history from
        
    Returns:
        Tuple of (history text, confidence score)
    """
    # Look for history patterns
    history_patterns = [
        r'(?:Examen effectué|Histoire clinique|Historique|Antécédents)[^\n\.]*?:?\s*([^\n]+)',
        r'(?:En complément|Complément)[^\n\.]*?:?\s*([^\n]+)'
    ]
    
    for pattern in history_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            history = match.group(1).strip()
            logger.debug(f"Extracted French history: {history}")
            return history, 0.85
    
    return "", 0.0

def extract_with_language_specific_patterns(text: str, language: str = None) -> Dict[str, Any]:
    """
    Extract information from text using language-specific patterns.
    
    Args:
        text: Text to extract information from
        language: Language code ('en', 'fr') or None for auto-detection
        
    Returns:
        Dictionary of extracted fields with their values and confidence scores
    """
    if not text:
        logger.warning("Empty text provided for extraction")
        return {}
    
    # Detect language if not specified
    if language is None:
        language = detect_language(text)
    
    logger.info(f"Extracting from {language} text")
    
    extracted_data = {}
    
    # Extract using language-specific patterns
    if language == "fr":
        # French extraction
        date_value, date_confidence = extract_date_french(text)
        if date_value:
            extracted_data['exam_date'] = {
                'value': date_value,
                'confidence': date_confidence
            }
        
        exam_type, exam_confidence = extract_exam_type_french(text)
        if exam_type:
            extracted_data['exam_type'] = {
                'value': exam_type,
                'confidence': exam_confidence
            }
        
        birads_result = extract_birads_french(text)
        if birads_result['value']:
            extracted_data['birads_score'] = birads_result
        
        # Extract provider information
        provider, provider_confidence = extract_provider_french(text)
        if provider:
            extracted_data['provider_name'] = {
                'value': provider,
                'confidence': provider_confidence
            }
        
        # Extract facility information
        facility, facility_confidence = extract_facility_french(text)
        if facility:
            extracted_data['facility'] = {
                'value': facility,
                'confidence': facility_confidence
            }
        
        # Extract history information
        history, history_confidence = extract_history_french(text)
        if history:
            extracted_data['history'] = {
                'value': history,
                'confidence': history_confidence
            }
        
        # Extract sections
        for section_name in ['findings', 'impression', 'recommendation']:
            section_text, section_confidence = extract_section_french(text, section_name)
            if section_text:
                extracted_data[section_name] = {
                    'value': section_text,
                    'confidence': section_confidence
                }
    
    # For English or other languages, we'll use the existing extractors
    # which will be called separately
    
    return extracted_data 