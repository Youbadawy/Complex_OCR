"""
Extraction pipeline for medical report text processing.

This module integrates all extraction components (language detection, 
spelling correction, multilingual extraction, and validation) into a 
comprehensive pipeline for processing medical report text.
"""

import logging
import json
from typing import Dict, Any, Optional, List

# Import our specialized modules
try:
    from src.document_processing.text_analysis import (
        preprocess_text_for_extraction,
        extract_birads_score,
        extract_provider_info, 
        extract_signed_by,
        extract_structured_data
    )
    from src.document_processing.multilingual_extraction import (
        detect_language,
        extract_with_language_specific_patterns
    )
    from src.document_processing.report_validation import (
        validate_and_enhance_extraction,
        cross_validate_extractions
    )
    # Set flag to indicate all imports succeeded
    ALL_MODULES_AVAILABLE = True
except ImportError as e:
    logging.error(f"Failed to import required modules: {str(e)}")
    ALL_MODULES_AVAILABLE = False

# Set up logger
logger = logging.getLogger(__name__)

class ExtractionPipeline:
    """
    Main extraction pipeline that orchestrates the entire extraction process.
    
    This pipeline handles language detection, preprocessing, extraction
    using language-specific patterns, cross-validation, and result enhancement.
    """
    
    def __init__(self):
        """Initialize the extraction pipeline."""
        if not ALL_MODULES_AVAILABLE:
            logger.warning("Some required modules are missing. Pipeline functionality will be limited.")
    
    def extract_from_text(self, text: str, deduplicate: bool = True) -> Dict[str, Any]:
        """
        Extract structured information from medical report text.
        
        Args:
            text: The raw text to extract information from
            deduplicate: Whether to deduplicate fields in the final output
            
        Returns:
            Dictionary of extracted fields
        """
        if not text:
            logger.warning("Empty text provided for extraction")
            return {}
        
        try:
            # Step 1: Detect language
            language = detect_language(text)
            logger.info(f"Detected language: {language}")
            
            # Step 2: Preprocess text (correct spelling, normalize whitespace, etc.)
            preprocessed_text = preprocess_text_for_extraction(text)
            logger.debug("Text preprocessing complete")
            
            # Step 3a: Extract with language-specific patterns
            language_specific_results = extract_with_language_specific_patterns(
                preprocessed_text, language
            )
            logger.debug(f"Language-specific extraction complete: {len(language_specific_results)} fields extracted")
            
            # Step 3b: Extract with standard patterns (our existing functions)
            standard_results = extract_structured_data(preprocessed_text)
            logger.debug(f"Standard extraction complete: {len(standard_results)} fields extracted")
            
            # Step 4: Cross-validate and merge results from both extraction methods
            merged_results = cross_validate_extractions(
                standard_results, language_specific_results
            )
            logger.debug(f"Cross-validation complete: {len(merged_results)} fields in final result")
            
            # Step 5: Apply final validation and enhancement
            enhanced_results = validate_and_enhance_extraction(merged_results)
            logger.debug("Validation and enhancement complete")
            
            # Step 6: Deduplicate fields if requested
            if deduplicate:
                enhanced_results = self._deduplicate_fields(enhanced_results)
                logger.debug("Field deduplication complete")
            
            return enhanced_results
            
        except Exception as e:
            logger.error(f"Error in extraction pipeline: {str(e)}")
            # Fall back to standard extraction if pipeline fails
            try:
                return extract_structured_data(text)
            except Exception as fallback_error:
                logger.error(f"Fallback extraction also failed: {str(fallback_error)}")
                return {}
    
    def _deduplicate_fields(self, extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Remove duplicate fields from the extracted data.
        
        Args:
            extracted_data: Dictionary of extracted fields
            
        Returns:
            Dictionary with duplicates removed
        """
        # Known duplicate field pairs
        duplicate_pairs = [
            # Common duplication patterns observed in the data
            ('impression', 'findings'),
            ('provider_name', 'signed_by'),
            ('facility', 'location')
        ]
        
        # Handle field pairs that might be duplicates
        for field1, field2 in duplicate_pairs:
            if field1 in extracted_data and field2 in extracted_data:
                # Get values, handling both direct values and dict objects
                val1 = extracted_data[field1].get('value', extracted_data[field1]) if isinstance(extracted_data[field1], dict) else extracted_data[field1]
                val2 = extracted_data[field2].get('value', extracted_data[field2]) if isinstance(extracted_data[field2], dict) else extracted_data[field2]
                
                # Check if they're identical or very similar
                if val1 == val2 or (isinstance(val1, str) and isinstance(val2, str) and self._text_similarity(val1, val2) > 0.9):
                    # Keep the one with higher confidence or the preferred field
                    conf1 = extracted_data[field1].get('confidence', 0.5) if isinstance(extracted_data[field1], dict) else 0.5
                    conf2 = extracted_data[field2].get('confidence', 0.5) if isinstance(extracted_data[field2], dict) else 0.5
                    
                    if conf1 >= conf2:
                        extracted_data.pop(field2, None)
                        logger.debug(f"Removed duplicate field '{field2}' (keeping '{field1}')")
                    else:
                        extracted_data.pop(field1, None)
                        logger.debug(f"Removed duplicate field '{field1}' (keeping '{field2}')")
        
        return extracted_data
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two text strings.
        
        Args:
            text1: First text string
            text2: Second text string
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        # Simple word overlap similarity
        if not text1 or not text2:
            return 0.0
            
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
            
        intersection = words1.intersection(words2)
        smaller_set = min(len(words1), len(words2))
        
        return len(intersection) / smaller_set if smaller_set > 0 else 0.0

def extract_all_fields_from_text(text: str) -> Dict[str, Any]:
    """
    Extract all available fields from the given medical report text.
    
    This is the main function to call from other parts of the application.
    
    Args:
        text: The raw OCR text from a medical report
        
    Returns:
        Dictionary containing all extracted fields
    """
    pipeline = ExtractionPipeline()
    return pipeline.extract_from_text(text) 