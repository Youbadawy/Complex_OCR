"""
Parser Integration Module

This module integrates the new MedicalReportParser with the existing application.
It provides compatibility functions to ensure the new parser works with the
existing UI and data processing pipeline.
"""

import logging
import pandas as pd
from typing import Dict, Any, List, Optional, Union
from pathlib import Path

# Import the new parser
from document_processing.medical_report_parser import MedicalReportParser, parse_medical_report

# Setup logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def process_ocr_text(text: str, use_llm: bool = False) -> Dict[str, Any]:
    """
    Process OCR text using the new parser, with compatibility for the existing system.
    
    Args:
        text: Raw OCR text extracted from a medical report
        use_llm: Flag to indicate if LLM enhancement should be used (not implemented yet)
        
    Returns:
        Dictionary with extracted fields in the format expected by the existing system
    """
    try:
        # Handle cases where text is not a string (e.g., if it's a dictionary or other data structure)
        if not isinstance(text, str):
            logger.warning(f"process_ocr_text received non-string input: {type(text)}")
            
            # If it's a dictionary, try to extract text from common keys
            if isinstance(text, dict):
                for key in ['text', 'raw_text', 'ocr_text', 'extracted_text']:
                    if key in text and isinstance(text[key], str):
                        text = text[key]
                        logger.info(f"Extracted text from dictionary key: {key}")
                        break
                else:
                    # Find any string that's likely the OCR text
                    for key, value in text.items():
                        if isinstance(value, str) and len(value) > 100:  # Arbitrary length check
                            text = value
                            logger.info(f"Extracted potential text from dictionary key: {key}")
                            break
                    else:
                        # If still no string found, convert to string representation
                        text = str(text)
                        logger.warning("No text field found, using string representation of the dictionary")
            else:
                # Convert to string if not a string or dictionary
                text = str(text)
                logger.warning("Non-string and non-dictionary input, using string representation")
        
        # Use the new parser
        parser = MedicalReportParser(text)
        extracted_fields = parser.extract_all_fields()
        
        # Convert to format expected by existing system
        # Ensure we get string values for database compatibility
        result = {}
        for field, item in extracted_fields.items():
            # Convert ExtractedField objects to strings
            if hasattr(item, 'value'):
                result[field] = item.value
            else:
                # If somehow not an ExtractedField, use directly
                result[field] = str(item) if not isinstance(item, str) else item
        
        # Add raw text for compatibility - this is important
        result['raw_ocr_text'] = text
        
        # Special handling for certain fields needed by the UI
        # Map clinical_history to patient_history for compatibility
        if 'clinical_history' in result and result['clinical_history'] != "N/A":
            result['patient_history'] = result['clinical_history']
        else:
            result['patient_history'] = "N/A"
        
        # Handle any special field mapping required by the UI
        # For example, ensure 'mammograph_results' is set based on 'findings'
        if 'findings' in result and result['findings'] != "N/A":
            result['mammograph_results'] = result['findings']
        else:
            result['mammograph_results'] = "N/A"
        
        # Ensure all values are strings for database compatibility
        for key in result:
            if isinstance(result[key], (dict, list, tuple)):
                result[key] = str(result[key])
            
        return result
        
    except Exception as e:
        logger.error(f"Error in process_ocr_text: {str(e)}")
        logger.exception("Full traceback:")
        # Return a minimal valid result to prevent UI errors
        return {
            'raw_ocr_text': text if isinstance(text, str) else str(text),
            'patient_name': "N/A",
            'age': "N/A",
            'exam_date': "N/A",
            'clinical_history': "N/A",
            'patient_history': "N/A",
            'findings': "N/A",
            'impression': "N/A",
            'recommendation': "N/A",
            'mammograph_results': "N/A",
            'birads_score': "N/A",
            'facility': "N/A",
            'exam_type': "N/A",
            'referring_provider': "N/A",
            'interpreting_provider': "N/A"
        }

def batch_process_texts(texts: List[str]) -> pd.DataFrame:
    """
    Process a batch of OCR texts and return a DataFrame.
    
    Args:
        texts: List of OCR texts to process
        
    Returns:
        DataFrame with extracted fields
    """
    # Create a parser for each text
    parsers = [MedicalReportParser(text) for text in texts]
    
    # Extract fields for each parser
    for parser in parsers:
        parser.extract_all_fields()
    
    # Convert to rows for DataFrame
    rows = []
    for parser in parsers:
        row_data = {field: item.value for field, item in parser.extracted_fields.items()}
        row_data['raw_ocr_text'] = parser.raw_text
        
        # Add compatibility fields
        if 'clinical_history' in row_data and row_data['clinical_history'] != "N/A":
            row_data['patient_history'] = row_data['clinical_history']
        if 'findings' in row_data and row_data['findings'] != "N/A":
            row_data['mammograph_results'] = row_data['findings']
            
        rows.append(row_data)
    
    # Create DataFrame
    return pd.DataFrame(rows)

def enhance_existing_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enhance an existing DataFrame by reprocessing raw OCR text with the new parser.
    
    Args:
        df: Existing DataFrame with 'raw_ocr_text' column
        
    Returns:
        Enhanced DataFrame with improved field extraction
    """
    if 'raw_ocr_text' not in df.columns:
        logger.warning("DataFrame does not contain 'raw_ocr_text' column, cannot enhance")
        return df
    
    # Create a copy to avoid modifying the original
    enhanced_df = df.copy()
    
    # Process each row
    for idx, row in df.iterrows():
        if isinstance(row['raw_ocr_text'], str) and row['raw_ocr_text']:
            # Extract fields using the new parser
            extracted = process_ocr_text(row['raw_ocr_text'])
            
            # Update row with extracted fields, preserving non-N/A values from original
            for field, value in extracted.items():
                if field in enhanced_df.columns:
                    # Only update if the new value is better (not N/A when original is N/A)
                    if value != "N/A" and (enhanced_df.at[idx, field] == "N/A" or 
                                           enhanced_df.at[idx, field] == "Not Available"):
                        enhanced_df.at[idx, field] = value
    
    return enhanced_df 