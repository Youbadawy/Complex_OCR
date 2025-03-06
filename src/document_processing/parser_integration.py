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
import traceback
import sys
import os
import re
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Attempt to fix import paths for different execution contexts
def ensure_src_in_path():
    """
    Ensure that the 'src' module can be imported by adding it to sys.path if needed.
    This handles cases where the code is run from different locations.
    """
    # Check if we can already import from src
    try:
        from src.document_processing.medical_report_parser import MedicalReportParser
        logger.info("Successfully imported MedicalReportParser, path is correct")
        return True
    except ImportError:
        # src is not in the path, try to add it
        logger.warning("Could not import from src, attempting to fix path")
        
        # Try different path possibilities
        current_file = Path(__file__).resolve()
        possible_src_paths = [
            # Current file's parent directory (if src/document_processing/parser_integration.py)
            current_file.parent.parent.parent,
            # Current file's parent's parent (if document_processing/parser_integration.py)
            current_file.parent.parent,
            # Current directory (if parser_integration.py with src as sibling)
            current_file.parent
        ]
        
        for path in possible_src_paths:
            if (path / "src").exists():
                logger.info(f"Found src directory at {path}")
                sys.path.insert(0, str(path))
                return True
            elif path.name == "src" and path.exists():
                logger.info(f"Found src directory at {path.parent}")
                sys.path.insert(0, str(path.parent))
                return True
        
        # If we couldn't find src, try adding the current working directory
        cwd = Path.cwd()
        sys.path.insert(0, str(cwd))
        logger.warning(f"Added current working directory to path: {cwd}")
        
        # Check if the fix worked
        try:
            from src.document_processing.medical_report_parser import MedicalReportParser
            logger.info("Successfully imported MedicalReportParser after path fix")
            return True
        except ImportError as e:
            logger.error(f"Still unable to import from src: {str(e)}")
            return False

# Try to fix the import path before importing MedicalReportParser
if ensure_src_in_path():
    try:
        from src.document_processing.medical_report_parser import MedicalReportParser, ExtractedField
        logger.info("Successfully imported MedicalReportParser")
    except ImportError as e:
        logger.error(f"Failed to import required modules: {str(e)}")
        # Define a fallback MedicalReportParser to avoid crashing
        class MedicalReportParser:
            def __init__(self, text):
                self.text = text
                self.language = "en"
                self.metadata = {"validation_issues": ["Parser module could not be loaded"]}
                self.extracted_fields = {}
                
            def extract_all_fields(self):
                return {"error": "Parser module could not be loaded"}
                
            def enhance_with_llm(self, llm_client=None, confidence_threshold=0.7):
                return self.extracted_fields
                
        class ExtractedField:
            def __init__(self, value, confidence=0.0, source="fallback"):
                self.value = value if value is not None else "N/A"
                self.confidence = confidence
                self.source = source

# Fallback LLM client implementation
class FallbackLLMClient:
    """
    A fallback LLM client that can be used when the real LLM client is not available.
    This provides basic functionality to improve extraction results using pattern matching
    and heuristics rather than a true LLM.
    """
    def __init__(self):
        self.logger = logging.getLogger(__name__ + ".FallbackLLMClient")
        self.logger.info("Using fallback LLM client - limited enhancement capabilities")
    
    def generate_text(self, prompt: str) -> str:
        """
        Generate text based on the prompt, using rule-based extraction
        instead of a real LLM.
        
        Args:
            prompt: The prompt containing extraction instructions and text
            
        Returns:
            A response formatted like an LLM would respond
        """
        self.logger.info("Generating text with fallback LLM client")
        
        # Extract the text from the prompt (between triple backticks)
        text_match = re.search(r'```\n(.*?)\n```', prompt, re.DOTALL)
        if not text_match:
            return "Error: Could not extract text from prompt"
        
        text = text_match.group(1)
        
        # Determine if this is a French or English document
        is_french = "français" in prompt.lower() or "en français" in prompt.lower()
        
        # Extract the fields to process from the prompt
        field_matches = re.findall(r'- ([A-Za-z_\s]+):', prompt)
        fields_to_process = [m.lower().replace(' ', '_') for m in field_matches]
        
        # Process each field with specialized heuristics
        response_lines = []
        
        for field in fields_to_process:
            if field == "clinical_history" or field == "patient_history":
                # Look for clinical history patterns
                patterns = [
                    # English patterns
                    r'(?:history|clinical history|indication)[:;]\s*(.*?)(?=\n\n|\n[A-Z]|$)',
                    # French patterns
                    r'(?:antécédents|histoire clinique|indication)[:;]\s*(.*?)(?=\n\n|\n[A-Z]|$)'
                ]
                value = self._extract_with_patterns(text, patterns, "clinical_history", is_french)
                response_lines.append(f"Clinical History: {value}")
                
            elif field == "findings" or field == "results":
                # Look for findings patterns
                patterns = [
                    # English patterns
                    r'(?:findings|results|observations)[:;]\s*(.*?)(?=\n\n|\n[A-Z]|$)',
                    # French patterns
                    r'(?:résultats|observations|constatations)[:;]\s*(.*?)(?=\n\n|\n[A-Z]|$)'
                ]
                value = self._extract_with_patterns(text, patterns, "findings", is_french)
                response_lines.append(f"Findings: {value}")
                
            elif field == "impression" or field == "conclusion":
                # Look for impression patterns
                patterns = [
                    # English patterns
                    r'(?:impression|conclusion|assessment)[:;]\s*(.*?)(?=\n\n|\n[A-Z]|$)',
                    # French patterns
                    r'(?:impression|conclusion|interprétation)[:;]\s*(.*?)(?=\n\n|\n[A-Z]|$)'
                ]
                value = self._extract_with_patterns(text, patterns, "impression", is_french)
                response_lines.append(f"Impression: {value}")
                
            elif field == "recommendation" or field == "follow_up":
                # Look for recommendation patterns
                patterns = [
                    # English patterns
                    r'(?:recommendation|follow.?up|plan)[:;]\s*(.*?)(?=\n\n|\n[A-Z]|$)',
                    # French patterns
                    r'(?:recommandation|suivi|conduite)[:;]\s*(.*?)(?=\n\n|\n[A-Z]|$)'
                ]
                value = self._extract_with_patterns(text, patterns, "recommendation", is_french)
                response_lines.append(f"Recommendation: {value}")
                
            elif field == "birads_score" or field == "birads":
                # Look for BIRADS patterns
                patterns = [
                    # English patterns
                    r'(?:BIRADS|BI-RADS|category)[:;]?\s*([0-6][A-Ca-c]?)',
                    # French patterns
                    r'(?:BIRADS|BI-RADS|catégorie)[:;]?\s*([0-6][A-Ca-c]?)'
                ]
                value = self._extract_with_patterns(text, patterns, "birads_score", is_french)
                response_lines.append(f"Birads Score: {value}")
                
            elif field == "exam_date":
                # Look for date patterns
                patterns = [
                    # YYYY-MM-DD
                    r'(?:exam|date)[:;]?\s*(\d{4}-\d{2}-\d{2})',
                    # DD/MM/YYYY or MM/DD/YYYY
                    r'(?:exam|date)[:;]?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{4})',
                    # Month names
                    r'(?:exam|date)[:;]?\s*(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4})'
                ]
                value = self._extract_with_patterns(text, patterns, "exam_date", is_french)
                response_lines.append(f"Exam Date: {value}")
            
            # Add more fields as needed with specialized extraction patterns
            
            else:
                # Generic field extraction (less accurate)
                response_lines.append(f"{field.replace('_', ' ').title()}: N/A")
        
        # Combine all lines for the response
        return "\n".join(response_lines)
    
    def _extract_with_patterns(self, text: str, patterns: List[str], field_type: str, is_french: bool) -> str:
        """Extract field values using regex patterns"""
        # Use appropriate patterns based on language
        if is_french:
            # Prioritize French patterns for French documents
            patterns = patterns[1:] + patterns[:1]
        
        # Try each pattern
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                value = match.group(1).strip()
                # Basic cleanup
                value = re.sub(r'\s+', ' ', value)
                value = value.strip()
                
                # If the value is too long, truncate it
                if len(value) > 200 and field_type not in ["findings", "impression"]:
                    value = value[:197] + "..."
                
                return value
        
        # No match found
        return "N/A"

# Function to get a fallback LLM client
def get_fallback_llm_client():
    """
    Get a fallback LLM client that can be used when the real LLM client is not available.
    
    Returns:
        A FallbackLLMClient instance
    """
    return FallbackLLMClient()

def process_ocr_text(text: str, llm_client=None, use_llm=False) -> Dict[str, Any]:
    """
    Process OCR text to extract structured information.
    
    Args:
        text: Raw OCR text from a medical document
        llm_client: Optional LLM client for enhancing extraction
        use_llm: Boolean flag for backward compatibility to toggle LLM enhancement
        
    Returns:
        Dictionary of extracted fields
    """
    # Input validation
    if not text:
        logging.warning("Empty text provided to process_ocr_text")
        return _create_empty_result(text)
    
    if not isinstance(text, str):
        logging.warning(f"Non-string input provided to process_ocr_text: {type(text)}")
        return _create_empty_result(text)
    
    try:
        # Create parser instance
        parser = MedicalReportParser(text)
        
        # Extract fields
        extracted_fields = parser.extract_all_fields()
        
        # Enhance with LLM if client is provided OR use_llm is True
        # This maintains backward compatibility with the use_llm flag
        should_use_llm = llm_client is not None or use_llm
        
        if should_use_llm:
            # If use_llm is True but no client is provided, try to use a default client
            actual_client = llm_client
            if actual_client is None:
                try:
                    # Try to import and use a default LLM client if available
                    try:
                        from src.llm_services.llm_client import get_default_llm_client
                        actual_client = get_default_llm_client()
                        logging.info("Using default LLM client for enhancement")
                    except (ImportError, AttributeError) as e:
                        logging.warning(f"Could not load default LLM client: {str(e)}")
                        # Use our fallback LLM client instead
                        actual_client = get_fallback_llm_client()
                        logging.info("Using fallback LLM client (rule-based) for enhancement")
                except Exception as e:
                    logging.warning(f"Error setting up any LLM client: {str(e)}")
                    actual_client = None
            
            # Apply LLM enhancement if we have a client
            if actual_client:
                enhanced_fields = parser.enhance_with_llm(actual_client, confidence_threshold=0.7)
                # Use enhanced fields if available
                if enhanced_fields:
                    extracted_fields = enhanced_fields
                    logging.info(f"LLM enhancement applied to fields: {parser.metadata.get('llm_enhancement', {}).get('fields_enhanced', [])}")
        
        # Convert to simple dictionary for database storage
        result = {}
        for field_name, field_obj in extracted_fields.items():
            # Ensure all values are strings for database compatibility
            if field_obj.value == "N/A":
                result[field_name] = ""
            else:
                result[field_name] = str(field_obj.value)
        
        # Map clinical_history to patient_history for backward compatibility
        if "clinical_history" in result:
            result["patient_history"] = result["clinical_history"]
        
        # Add raw text for compatibility with existing code
        result["raw_ocr_text"] = text
        
        # Add metadata for debugging
        result["_extraction_metadata"] = {
            "language": parser.language,
            "confidence_scores": {k: v.confidence for k, v in extracted_fields.items()},
            "extraction_sources": {k: v.source for k, v in extracted_fields.items()},
            "validation_issues": parser.metadata.get("validation_issues", []),
            "llm_enhanced": should_use_llm and "llm_enhancement" in parser.metadata
        }
        
        return result
    
    except Exception as e:
        logging.error(f"Error in process_ocr_text: {str(e)}", exc_info=True)
        return _create_empty_result(text)

def _create_empty_result(text: Any) -> Dict[str, Any]:
    """
    Create an empty result dictionary when extraction fails.
    
    Args:
        text: The input text that failed to process
        
    Returns:
        Dictionary with empty values for all expected fields
    """
    # Return a dictionary with default "N/A" values
    result = {
        "patient_name": "Unknown Patient",
        "age": "",
        "clinical_history": "",
        "patient_history": "",  # Alias for clinical_history
        "findings": "",
        "impression": "",
        "recommendation": "",
        "birads_score": "",
        "exam_date": "",
        "facility": "",
        "exam_type": "",
        "referring_provider": "",
        "interpreting_provider": ""
    }
    
    # Add raw text for debugging
    if isinstance(text, str):
        result["raw_ocr_text"] = text
    else:
        result["raw_ocr_text"] = str(text)
    
    # Add metadata indicating failure
    result["_extraction_metadata"] = {
        "error": True,
        "validation_issues": ["Extraction failed completely"],
        "language": "unknown"
    }
    
    return result

def is_valid_age(age_str: str) -> bool:
    """
    Check if a string represents a valid age.
    
    Args:
        age_str: String representing an age
        
    Returns:
        True if the string represents a valid age (18-120), False otherwise
    """
    try:
        # Convert to integer
        age = int(age_str)
        # Check if in reasonable range (18-120)
        return 18 <= age <= 120
    except (ValueError, TypeError):
        return False

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

def debug_parser(text: str, llm_client=None, use_llm=False) -> Dict[str, Any]:
    """
    Run the parser with detailed debug information.
    
    Args:
        text: Raw OCR text from a medical document
        llm_client: Optional LLM client for enhancing extraction
        use_llm: Boolean flag for backward compatibility to toggle LLM enhancement
        
    Returns:
        Dictionary with extraction results, metadata, and text statistics
    """
    try:
        # Create parser instance
        parser = MedicalReportParser(text)
        
        # Extract fields
        extracted_fields = parser.extract_all_fields()
        
        # Enhance with LLM if client is provided or use_llm is True
        llm_enhanced_fields = None
        should_use_llm = llm_client is not None or use_llm
        llm_client_type = "none"
        
        if should_use_llm:
            # If use_llm is True but no client is provided, try to use a default client
            actual_client = llm_client
            if actual_client is None:
                try:
                    # Try to import and use a default LLM client if available
                    try:
                        from src.llm_services.llm_client import get_default_llm_client
                        actual_client = get_default_llm_client()
                        llm_client_type = "default"
                        logging.info("Using default LLM client for enhancement in debug_parser")
                    except (ImportError, AttributeError) as e:
                        logging.warning(f"Could not load default LLM client in debug_parser: {str(e)}")
                        # Use our fallback LLM client instead
                        actual_client = get_fallback_llm_client()
                        llm_client_type = "fallback"
                        logging.info("Using fallback LLM client (rule-based) for enhancement in debug_parser")
                except Exception as e:
                    logging.warning(f"Error setting up any LLM client in debug_parser: {str(e)}")
                    actual_client = None
            else:
                llm_client_type = "provided"
            
            # Apply LLM enhancement if we have a client
            if actual_client:
                llm_enhanced_fields = parser.enhance_with_llm(actual_client, confidence_threshold=0.7)
        
        # Prepare debug output
        debug_info = {
            "rule_based_extraction": {
                k: {
                    "value": v.value,
                    "confidence": v.confidence,
                    "source": v.source
                } for k, v in extracted_fields.items()
            },
            "metadata": parser.metadata,
            "text_stats": {
                "length": len(text),
                "line_count": text.count('\n') + 1,
                "language": parser.language,
                "llm_client_type": llm_client_type
            }
        }
        
        # Add LLM enhanced results if available
        if llm_enhanced_fields:
            debug_info["llm_enhanced_extraction"] = {
                k: {
                    "value": v.value,
                    "confidence": v.confidence,
                    "source": v.source
                } for k, v in llm_enhanced_fields.items()
            }
            
            # Show differences between rule-based and LLM-enhanced
            debug_info["field_differences"] = {}
            for field_name in parser.EXPECTED_FIELDS:
                if field_name in extracted_fields and field_name in llm_enhanced_fields:
                    rule_value = extracted_fields[field_name].value
                    llm_value = llm_enhanced_fields[field_name].value
                    if rule_value != llm_value:
                        debug_info["field_differences"][field_name] = {
                            "rule_based": rule_value,
                            "llm_enhanced": llm_value
                        }
        
        return debug_info
    
    except Exception as e:
        logging.error(f"Error in debug_parser: {str(e)}", exc_info=True)
        return {"error": f"Debug error: {str(e)}"} 