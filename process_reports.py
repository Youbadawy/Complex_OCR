#!/usr/bin/env python3
"""
Command-line interface for processing medical reports.

This script provides a command-line interface for processing medical reports,
extracting structured information, and validating results.
"""

import os
import sys
import argparse
import logging
import pandas as pd
from typing import Dict, Any, List, Optional
import traceback

# Add parent directory to path to import from src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("processing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import main processor functions
try:
    from src.document_processing.main_processor import (
        process_single_report,
        process_directory,
        fix_dataframe
    )
    # Check if enhanced pipeline is available
    try:
        from src.document_processing import ENHANCED_PIPELINE_AVAILABLE
        logger.info(f"Enhanced extraction pipeline available: {ENHANCED_PIPELINE_AVAILABLE}")
    except ImportError:
        logger.warning("Enhanced extraction pipeline not available")
        ENHANCED_PIPELINE_AVAILABLE = False
except ImportError as e:
    logger.error(f"Failed to import processor functions: {str(e)}")
    sys.exit(1)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Process medical reports and extract structured information.")
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("-i", "--input", help="Path to a single PDF file to process")
    input_group.add_argument("-d", "--directory", help="Path to a directory of PDF files to process")
    input_group.add_argument("-f", "--fix", help="Path to an existing output CSV to fix")
    
    # Output options
    parser.add_argument("-o", "--output", help="Path to output CSV file (default: output.csv)", default="output.csv")
    
    # Processing options
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM-based enhancements")
    parser.add_argument("--no-pipeline", action="store_true", help="Disable enhanced extraction pipeline")
    
    # Debugging options
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    
    return parser.parse_args()

def process_file(file_path: str, use_llm: bool = True, use_pipeline: bool = True) -> Dict[str, Any]:
    """Process a single file."""
    try:
        logger.info(f"Processing file: {file_path}")
        # Process file using either enhanced pipeline or standard methods
        result = process_single_report(file_path, use_llm)
        
        # Remove duplicate columns if they exist
        result = deduplicate_fields(result)
        
        logger.info(f"Successfully processed {file_path}")
        return result
    except Exception as e:
        logger.error(f"Error processing {file_path}: {str(e)}")
        return {
            'source_file': os.path.basename(file_path),
            'processing_status': 'error',
            'error_message': str(e)
        }

def deduplicate_fields(data: Dict[str, Any]) -> Dict[str, Any]:
    """Remove duplicate fields from extracted data."""
    # List of known field pairs that could be duplicates
    duplicate_pairs = [
        # Format: (main_field, duplicate_field)
        ('provider_info.referring_provider', 'referring_provider'),
        ('provider_info.interpreting_provider', 'interpreting_provider'),
        ('provider_info.facility', 'facility'),
        ('clinical_history', 'history')
    ]
    
    deduplicated = data.copy()
    
    # Check for provider_info dictionary and extract its fields if needed
    if 'provider_info' in deduplicated and isinstance(deduplicated['provider_info'], dict):
        provider_info = deduplicated['provider_info']
        
        # Flatten provider_info for easier processing
        for key, value in provider_info.items():
            if value:  # Only add non-empty values
                flat_key = f"provider_info.{key}"
                deduplicated[flat_key] = value
    
    # Remove known duplicate fields, keeping the one with content
    for main_field, duplicate_field in duplicate_pairs:
        if main_field in deduplicated and duplicate_field in deduplicated:
            # Keep the field with content, prioritizing the main field
            if deduplicated[main_field] and not deduplicated[duplicate_field]:
                del deduplicated[duplicate_field]
            elif not deduplicated[main_field] and deduplicated[duplicate_field]:
                deduplicated[main_field] = deduplicated[duplicate_field]
                del deduplicated[duplicate_field]
            else:
                # Both have content or are empty - keep main field
                del deduplicated[duplicate_field]
    
    return deduplicated

def process_dir(directory_path: str, output_file: str, use_llm: bool = True, use_pipeline: bool = True) -> None:
    """Process all PDF files in a directory."""
    try:
        logger.info(f"Processing directory: {directory_path}")
        
        # Use the built-in directory processor
        df = process_directory(directory_path, output_file, use_llm)
        
        # Deduplicate columns in the dataframe
        deduplicated_records = []
        for _, row in df.iterrows():
            deduplicated_records.append(deduplicate_fields(row.to_dict()))
        
        # Create a new dataframe with deduplicated data
        deduplicated_df = pd.DataFrame(deduplicated_records)
        
        # Save to CSV
        deduplicated_df.to_csv(output_file, index=False)
        logger.info(f"Successfully processed directory. Results saved to {output_file}")
        
        # Print a summary
        successful = len(deduplicated_df[deduplicated_df['processing_status'] == 'success'])
        total = len(deduplicated_df)
        logger.info(f"Processed {successful}/{total} files successfully")
        
    except Exception as e:
        logger.error(f"Error processing directory {directory_path}: {str(e)}")
        logger.error(traceback.format_exc())

def fix_existing_dataframe(input_file: str, output_file: str) -> None:
    """Fix and enhance an existing output dataframe."""
    try:
        logger.info(f"Fixing existing dataframe: {input_file}")
        
        # Read the existing dataframe
        df = pd.read_csv(input_file)
        logger.info(f"Read dataframe with {len(df)} rows and {len(df.columns)} columns")
        
        # Apply fixes using the fix_dataframe function
        fixed_df = fix_dataframe(df)
        
        # Deduplicate columns
        deduplicated_records = []
        for _, row in fixed_df.iterrows():
            deduplicated_records.append(deduplicate_fields(row.to_dict()))
        
        # Create a new dataframe with deduplicated data
        deduplicated_df = pd.DataFrame(deduplicated_records)
        
        # Save to CSV
        deduplicated_df.to_csv(output_file, index=False)
        logger.info(f"Successfully fixed dataframe. Results saved to {output_file}")
        
    except Exception as e:
        logger.error(f"Error fixing dataframe {input_file}: {str(e)}")
        logger.error(traceback.format_exc())

def main():
    """Main entry point."""
    args = parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")
    
    # Disable LLM-based enhancements if requested
    use_llm = not args.no_llm
    if not use_llm:
        logger.info("LLM-based enhancements disabled")
    
    # Disable enhanced pipeline if requested
    use_pipeline = not args.no_pipeline and ENHANCED_PIPELINE_AVAILABLE
    if not use_pipeline:
        logger.info("Enhanced extraction pipeline disabled")
    
    # Process input based on the provided arguments
    if args.input:
        # Process a single file
        result = process_file(args.input, use_llm, use_pipeline)
        
        # Convert to dataframe and save
        df = pd.DataFrame([result])
        df.to_csv(args.output, index=False)
        logger.info(f"Results saved to {args.output}")
        
    elif args.directory:
        # Process a directory
        process_dir(args.directory, args.output, use_llm, use_pipeline)
        
    elif args.fix:
        # Fix an existing output file
        fix_existing_dataframe(args.fix, args.output)
    
    logger.info("Processing complete")

if __name__ == "__main__":
    main() 