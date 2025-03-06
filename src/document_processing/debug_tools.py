"""
Debugging tools for OCR and extraction processes.

This module provides utilities to help diagnose issues with OCR text extraction
and medical report parsing.
"""

import logging
import json
import inspect
import pandas as pd
import traceback
from typing import Any, Dict, List, Tuple, Optional, Union

# Setup logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def inspect_ocr_result(result: Any) -> Dict[str, Any]:
    """
    Inspect an OCR result to understand its structure and content.
    
    Args:
        result: The OCR result returned by process_pdf
        
    Returns:
        Dictionary with debug information about the OCR result
    """
    debug_info = {
        'type': str(type(result)),
        'contents': {}
    }
    
    # Inspect by type
    if isinstance(result, dict):
        debug_info['keys'] = list(result.keys())
        
        # Check contents of each key
        for key, value in result.items():
            debug_info['contents'][key] = {
                'type': str(type(value)),
                'is_empty': value is None or (hasattr(value, '__len__') and len(value) == 0),
                'sample': str(value)[:200] + '...' if isinstance(value, str) and len(str(value)) > 200 else str(value)
            }
            
    elif isinstance(result, tuple) or isinstance(result, list):
        debug_info['length'] = len(result)
        
        # Inspect each element
        for i, item in enumerate(result):
            debug_info['contents'][f'element_{i}'] = {
                'type': str(type(item)),
                'is_empty': item is None or (hasattr(item, '__len__') and len(item) == 0),
            }
            
            # For dictionary elements, show keys
            if isinstance(item, dict):
                debug_info['contents'][f'element_{i}']['keys'] = list(item.keys())
                
                # Inspect content of each key
                for key, value in item.items():
                    debug_info['contents'][f'element_{i}'][f'key_{key}'] = {
                        'type': str(type(value)),
                        'is_empty': value is None or (hasattr(value, '__len__') and len(value) == 0),
                        'sample': str(value)[:200] + '...' if isinstance(value, str) and len(str(value)) > 200 else str(value)
                    }
    else:
        # For other types, just show string representation
        debug_info['string_value'] = str(result)[:1000] + '...' if len(str(result)) > 1000 else str(result)
    
    return debug_info

def extract_text_from_ocr_result(result: Any) -> Tuple[Optional[str], str]:
    """
    Extract text from an OCR result, trying multiple potential structures.
    
    Args:
        result: The OCR result returned by process_pdf
        
    Returns:
        Tuple of (extracted_text, debug_message)
    """
    debug_msg = []
    
    # Try various extraction methods
    if result is None:
        debug_msg.append("OCR result is None")
        return None, "\n".join(debug_msg)
    
    # Case 1: result is a dictionary with text field
    if isinstance(result, dict):
        debug_msg.append(f"OCR result is a dictionary with keys: {list(result.keys())}")
        
        # Try common text keys
        for key in ['text', 'raw_text', 'ocr_text', 'extracted_text', 'content']:
            if key in result and isinstance(result[key], str) and result[key].strip():
                debug_msg.append(f"Found text in key '{key}' ({len(result[key])} characters)")
                return result[key], "\n".join(debug_msg)
        
        # Try any string that looks like OCR text
        for key, value in result.items():
            if isinstance(value, str) and len(value) > 100:  # Arbitrary length check
                debug_msg.append(f"Found potential text in key '{key}' ({len(value)} characters)")
                return value, "\n".join(debug_msg)
    
    # Case 2: result is a tuple with dictionaries
    elif isinstance(result, tuple) or isinstance(result, list):
        debug_msg.append(f"OCR result is a {'tuple' if isinstance(result, tuple) else 'list'} with {len(result)} elements")
        
        # First check if any element is a string
        for i, item in enumerate(result):
            if isinstance(item, str) and len(item) > 100:
                debug_msg.append(f"Found text in element {i} ({len(item)} characters)")
                return item, "\n".join(debug_msg)
        
        # Then check for dictionaries
        for i, item in enumerate(result):
            if isinstance(item, dict):
                debug_msg.append(f"Element {i} is a dictionary with keys: {list(item.keys())}")
                
                # Try common text keys
                for key in ['text', 'raw_text', 'ocr_text', 'extracted_text', 'content']:
                    if key in item and isinstance(item[key], str) and item[key].strip():
                        debug_msg.append(f"Found text in element {i}, key '{key}' ({len(item[key])} characters)")
                        return item[key], "\n".join(debug_msg)
                
                # Try any string that looks like OCR text
                for key, value in item.items():
                    if isinstance(value, str) and len(value) > 100:
                        debug_msg.append(f"Found potential text in element {i}, key '{key}' ({len(value)} characters)")
                        return value, "\n".join(debug_msg)
    
    # Case 3: result is a string
    elif isinstance(result, str) and result.strip():
        debug_msg.append(f"OCR result is a string ({len(result)} characters)")
        return result, "\n".join(debug_msg)
    
    # No text found
    debug_msg.append("No text could be extracted from OCR result")
    return None, "\n".join(debug_msg)

def test_extract_text(ocr_function, test_file):
    """
    Test extracting text from a file using the OCR function.
    
    Args:
        ocr_function: Function to perform OCR
        test_file: File to test with
        
    Returns:
        Dictionary with test results and debug information
    """
    try:
        # Run OCR function
        ocr_result = ocr_function(test_file)
        
        # Inspect result
        debug_info = inspect_ocr_result(ocr_result)
        
        # Try to extract text
        extracted_text, debug_msg = extract_text_from_ocr_result(ocr_result)
        
        return {
            'success': extracted_text is not None,
            'extracted_text': extracted_text,
            'debug_message': debug_msg,
            'debug_info': debug_info
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        } 