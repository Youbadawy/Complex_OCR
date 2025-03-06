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
from .pdf_processor import process_pdf, extract_text_from_pdf
from .text_analysis import process_document_text
from .report_validation import validate_report_data, extract_and_validate_report

# Check if LLM integration is available
try:
    from .llm_extraction import extract_with_llm, LLM_AVAILABLE
except ImportError:
    logger.warning("LLM extraction module not available")
    LLM_AVAILABLE = False

class ProcessingError(Exception):
    """Base exception for processing errors"""
    pass

class PDFProcessingError(ProcessingError):
    """Exception raised when PDF processing fails"""
    pass

class TextProcessingError(ProcessingError):
    """Exception raised when text processing fails"""
    pass

class ValidationError(ProcessingError):
    """Exception raised when validation fails"""
    pass

def process_single_report(pdf_path: str, use_llm: bool = True) -> Dict[str, Any]:
    """
    Process a single medical report from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        use_llm: Whether to use LLM enhancement if available
        
    Returns:
        Dictionary with extracted and validated report data
        
    Raises:
        ProcessingError: If processing fails
    """
    try:
        logger.info(f"Processing report: {pdf_path}")
        
        # Step 1: Extract text from PDF
        try:
            raw_text = extract_text_from_pdf(pdf_path)
            if not raw_text:
                logger.warning(f"No text extracted from {pdf_path}")
                raise PDFProcessingError(f"No text extracted from {pdf_path}")
            logger.info(f"Extracted {len(raw_text)} characters from PDF")
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            raise PDFProcessingError(f"Error extracting text from PDF: {str(e)}")
        
        # Step 2: Process the raw text using conventional methods
        try:
            extracted_data = process_document_text(raw_text)
            logger.info("Basic extraction completed")
        except Exception as e:
            logger.error(f"Error in text processing: {str(e)}")
            raise TextProcessingError(f"Error in text processing: {str(e)}")
        
        # Step 3: Validate and enhance the extracted data
        try:
            validated_data = validate_report_data(extracted_data, raw_text)
            logger.info("Validation completed")
        except Exception as e:
            logger.error(f"Error in validation: {str(e)}")
            raise ValidationError(f"Error in validation: {str(e)}")
        
        # Step 4: Use LLM for missing or low-confidence fields if available
        if use_llm and LLM_AVAILABLE:
            try:
                # Check for critical missing fields or low confidence
                critical_fields = ["birads_score", "impression", "exam_date", "exam_type"]
                missing_fields = [field for field in critical_fields 
                                if field not in validated_data or 
                                validated_data[field] in ["", "Not Available"]]
                
                if missing_fields:
                    logger.info(f"Using LLM to extract missing fields: {missing_fields}")
                    llm_data = extract_with_llm(raw_text, "full")
                    
                    # Update missing fields from LLM data
                    for field in missing_fields:
                        if field in llm_data and llm_data[field]:
                            validated_data[field] = llm_data[field]
                            logger.info(f"Updated {field} with LLM data")
                    
                    # Re-validate with the new data
                    validated_data = validate_report_data(validated_data, raw_text)
            except Exception as e:
                logger.warning(f"LLM enhancement failed, using conventional results: {str(e)}")
        
        # Step 5: Add additional metadata
        validated_data['source_file'] = os.path.basename(pdf_path)
        validated_data['processing_status'] = 'success'
        
        return validated_data
    
    except ProcessingError as e:
        # Return partial data with error status
        return {
            'source_file': os.path.basename(pdf_path),
            'processing_status': 'error',
            'error_message': str(e)
        }
    except Exception as e:
        # Catch any unexpected errors
        error_msg = f"Unexpected error processing {pdf_path}: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return {
            'source_file': os.path.basename(pdf_path),
            'processing_status': 'error',
            'error_message': error_msg
        }

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
    Fix and enhance a dataframe of report data.
    
    Args:
        df: DataFrame with extracted report data
        raw_texts: Optional dictionary mapping source_file to raw text
        
    Returns:
        Enhanced DataFrame with fixed inconsistencies
    """
    logger.info("Fixing dataframe inconsistencies")
    
    # Create a copy to avoid modifying the original
    fixed_df = df.copy()
    
    # Check for duplicate columns
    duplicate_cols = fixed_df.columns[fixed_df.columns.duplicated()]
    if len(duplicate_cols) > 0:
        logger.warning(f"Found duplicate columns: {duplicate_cols}")
        for col in duplicate_cols:
            # Keep the first instance of the column and drop duplicates
            fixed_df = fixed_df.loc[:,~fixed_df.columns.duplicated()]
    
    # Fix inconsistencies in each row if we have raw text
    if raw_texts:
        for idx, row in fixed_df.iterrows():
            if 'source_file' in row and row['source_file'] in raw_texts:
                source_file = row['source_file']
                raw_text = raw_texts[source_file]
                
                # Create a record from the row
                record = row.to_dict()
                
                try:
                    # Validate and enhance the record
                    fixed_record = validate_report_data(record, raw_text)
                    
                    # Update the dataframe row
                    for key, value in fixed_record.items():
                        if key in fixed_df.columns:
                            fixed_df.at[idx, key] = value
                except Exception as e:
                    logger.error(f"Error fixing row {idx}: {str(e)}")
    
    # Normalize column values
    if 'birads_score' in fixed_df.columns:
        try:
            from .report_validation import normalize_birads
            fixed_df['birads_score'] = fixed_df['birads_score'].apply(
                lambda x: normalize_birads(x) if isinstance(x, str) else x
            )
        except Exception as e:
            logger.error(f"Error normalizing BIRADS scores: {str(e)}")
    
    if 'exam_date' in fixed_df.columns:
        try:
            from .report_validation import normalize_date
            fixed_df['exam_date'] = fixed_df['exam_date'].apply(
                lambda x: normalize_date(x) if isinstance(x, str) else x
            )
        except Exception as e:
            logger.error(f"Error normalizing dates: {str(e)}")
    
    if 'exam_type' in fixed_df.columns:
        try:
            from .report_validation import normalize_exam_type
            fixed_df['exam_type'] = fixed_df['exam_type'].apply(
                lambda x: normalize_exam_type(x) if isinstance(x, str) else x
            )
        except Exception as e:
            logger.error(f"Error normalizing exam types: {str(e)}")
    
    return fixed_df

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