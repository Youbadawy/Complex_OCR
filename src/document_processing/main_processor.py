"""
Main processor module for Medical Report Processor application.

This module integrates all components of the Medical Report Processor
to provide a complete processing pipeline for medical reports.
"""

import os
import logging
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
import traceback

# Setup logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import processing modules
from .pdf_processor import process_pdf
from .text_analysis import process_document_text
from .report_validation import validate_report_data, extract_and_validate_report
from document_processing.ocr import is_deidentified_document

# Import our enhanced extraction pipeline
try:
    from .extraction_pipeline import ExtractionPipeline, extract_all_fields_from_text
    from .multilingual_extraction import detect_language
    PIPELINE_AVAILABLE = True
    logger.info("Enhanced extraction pipeline is available")
except ImportError as e:
    logger.warning(f"Enhanced extraction pipeline not available: {str(e)}")
    PIPELINE_AVAILABLE = False

# Check if LLM integration is available
try:
    from .llm_extraction import extract_with_llm, LLM_AVAILABLE
except ImportError:
    logger.warning("LLM extraction module not available")
    LLM_AVAILABLE = False

# Import the new parser integration
from document_processing.parser_integration import process_ocr_text, enhance_existing_dataframe

class ProcessingError(Exception):
    """Base exception for document processing errors"""
    pass

class PDFProcessingError(ProcessingError):
    """Exception raised for errors during PDF processing"""
    pass

class TextProcessingError(ProcessingError):
    """Exception raised for errors during text extraction or processing"""
    pass

class ValidationError(ProcessingError):
    """Exception raised for validation errors"""
    pass

def process_single_report(pdf_path: str, use_llm: bool = True) -> Dict[str, Any]:
    """
    Process a single medical report PDF and extract structured data.
    
    Args:
        pdf_path: Path to the PDF file
        use_llm: Whether to use LLM for enhancing extraction
        
    Returns:
        Dictionary with extracted structured data
    """
    logger.info(f"Processing report: {pdf_path}")
    
    try:
        # Extract text and perform OCR
        ocr_result = process_pdf(pdf_path)
        
        if not ocr_result or not ocr_result.get('text'):
            raise TextProcessingError(f"Failed to extract text from {pdf_path}")
        
        raw_text = ocr_result.get('text', '')
        
        # Check if the document appears to be de-identified
        is_deidentified = is_deidentified_document(raw_text)
        
        # Use the new parser to extract structured data
        extracted_data = process_ocr_text(raw_text, use_llm=use_llm)
        
        # Add metadata
        extracted_data['file_path'] = pdf_path
        extracted_data['file_name'] = os.path.basename(pdf_path)
        extracted_data['is_deidentified'] = is_deidentified
        
        return extracted_data
        
    except Exception as e:
        error_msg = f"Error processing {pdf_path}: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        raise ProcessingError(error_msg) from e

def process_directory(directory_path: str, output_file: str = None, use_llm: bool = True) -> pd.DataFrame:
    """
    Process all PDF files in a directory.
    
    Args:
        directory_path: Path to directory containing PDF files
        output_file: Optional path to save output DataFrame as CSV
        use_llm: Whether to use LLM enhancement if available
        
    Returns:
        DataFrame with extracted and validated data from all reports
        
    Raises:
        ProcessingError: If directory processing fails
    """
    logger.info(f"Processing directory: {directory_path}")
    
    # Check if directory exists
    if not os.path.isdir(directory_path):
        raise ProcessingError(f"Directory not found: {directory_path}")
    
    # Get all PDF files in the directory
    pdf_files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) 
                if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        logger.warning(f"No PDF files found in {directory_path}")
        return pd.DataFrame()
    
    logger.info(f"Found {len(pdf_files)} PDF files to process")
    
    # Process each PDF file
    results = []
    for pdf_file in pdf_files:
        try:
            result = process_single_report(pdf_file, use_llm)
            results.append(result)
            logger.info(f"Processed {pdf_file}")
        except Exception as e:
            logger.error(f"Error processing {pdf_file}: {str(e)}")
            results.append({
                'source_file': os.path.basename(pdf_file),
                'processing_status': 'error',
                'error_message': str(e)
            })
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Save to CSV if output file specified
    if output_file:
        try:
            df.to_csv(output_file, index=False)
            logger.info(f"Results saved to {output_file}")
        except Exception as e:
            logger.error(f"Error saving results to {output_file}: {str(e)}")
    
    return df

def fix_dataframe(df: pd.DataFrame, raw_texts: Dict[str, str] = None) -> pd.DataFrame:
    """
    Enhance and fix issues in the extracted dataframe.
    
    Args:
        df: DataFrame with extracted structured data
        raw_texts: Optional dictionary of raw OCR texts, keyed by file path
        
    Returns:
        Enhanced DataFrame with improved extractions
    """
    if df.empty:
        return df
        
    # Make a copy to avoid modifying the original
    enhanced_df = df.copy()
    
    # Replace "Not Available" with "N/A" for consistency
    enhanced_df = enhanced_df.applymap(
        lambda x: "N/A" if x == "Not Available" or (isinstance(x, str) and x.strip() == "") else x
    )
    
    # Use the new parser integration to enhance extraction
    if 'raw_ocr_text' in enhanced_df.columns:
        enhanced_df = enhance_existing_dataframe(enhanced_df)
    elif raw_texts:
        # If raw texts are provided separately, add them to the dataframe
        for idx, row in enhanced_df.iterrows():
            file_path = row.get('file_path', '')
            if file_path in raw_texts:
                enhanced_df.at[idx, 'raw_ocr_text'] = raw_texts[file_path]
        
        # Then enhance with the new parser
        enhanced_df = enhance_existing_dataframe(enhanced_df)
    
    # Standardize columns
    required_columns = [
        'patient_name', 'age', 'exam_date', 
        'clinical_history', 'patient_history',
        'findings', 'impression', 'recommendation', 
        'mammograph_results', 'birads_score',
        'facility', 'exam_type',
        'referring_provider', 'interpreting_provider',
        'raw_ocr_text'
    ]
    
    # Ensure all required columns exist
    for col in required_columns:
        if col not in enhanced_df.columns:
            enhanced_df[col] = "N/A"
    
    # Fix redundant fields - if two fields contain the same information, keep just one
    # Map patient_history to clinical_history or vice versa
    enhanced_df = _handle_redundant_fields(enhanced_df, 'patient_history', 'clinical_history')
    
    # Map findings to mammograph_results or vice versa
    enhanced_df = _handle_redundant_fields(enhanced_df, 'findings', 'mammograph_results')
    
    # Map impression to findings if findings is empty
    enhanced_df = _handle_redundant_fields(enhanced_df, 'impression', 'findings')
    
    return enhanced_df

def _handle_redundant_fields(df: pd.DataFrame, field1: str, field2: str) -> pd.DataFrame:
    """
    Handle redundant fields in the dataframe by consolidating their values.
    
    Args:
        df: DataFrame to process
        field1: First field name
        field2: Second field name
        
    Returns:
        DataFrame with consolidated fields
    """
    if field1 not in df.columns or field2 not in df.columns:
        return df
        
    for idx, row in df.iterrows():
        val1 = row[field1]
        val2 = row[field2]
        
        # If one is N/A but the other isn't, fill the N/A one
        if val1 == "N/A" and val2 != "N/A":
            df.at[idx, field1] = val2
        elif val2 == "N/A" and val1 != "N/A":
            df.at[idx, field2] = val1
            
    return df

def main():
    """Main function to demonstrate the module's usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Process medical reports.')
    parser.add_argument('input', help='Input PDF file or directory containing PDF files')
    parser.add_argument('--output', '-o', help='Output CSV file')
    parser.add_argument('--no-llm', action='store_true', help='Disable LLM enhancement')
    
    args = parser.parse_args()
    
    if os.path.isdir(args.input):
        df = process_directory(args.input, args.output, not args.no_llm)
        print(f"Processed {len(df)} reports")
    elif os.path.isfile(args.input):
        result = process_single_report(args.input, not args.no_llm)
        df = pd.DataFrame([result])
        if args.output:
            df.to_csv(args.output, index=False)
        print(df)
    else:
        print(f"Input not found: {args.input}")

if __name__ == "__main__":
    main() 