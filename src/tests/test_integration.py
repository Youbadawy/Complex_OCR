"""
Integration tests for the Medical Report Processor.

This script tests the full extraction and validation pipeline
to ensure all components work together correctly.
"""

import sys
import os
import logging
from pprint import pprint

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import the modules we want to test
try:
    from src.document_processing.text_analysis import extract_birads_score, process_document_text
    from src.document_processing.report_validation import validate_report_data
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

def run_tests():
    """Run integration tests for the Medical Report Processor."""
    print("\n===== TESTING EXTRACTION AND VALIDATION PIPELINE =====\n")
    
    # Sample report text
    sample_report = """
    MAMMOGRAM REPORT
    
    EXAM DATE: 01/15/2023
    PATIENT: JANE DOE
    
    CLINICAL HISTORY:
    57-year-old female presenting for routine screening mammogram.
    
    TECHNIQUE:
    Standard four-view mammogram was performed.
    
    FINDINGS:
    The breast parenchyma is heterogeneously dense, which may obscure small masses.
    There is a new 8mm spiculated mass in the upper outer quadrant of the right breast
    at the 10 o'clock position, 4cm from the nipple. This was not present on prior 
    examinations and is suspicious for malignancy.
    
    No suspicious calcifications, skin thickening, or nipple retraction are seen.
    
    Left breast: No suspicious masses, calcifications, or architectural distortion.
    
    IMPRESSION:
    New 8mm spiculated mass in the right breast at 10 o'clock position, 4cm from nipple.
    This is highly suspicious for malignancy.
    BLRADS 4c
    
    RECOMMENDATION:
    Ultrasound-guided core needle biopsy of the right breast mass is recommended.
    Please contact our department to schedule this procedure.
    
    RADIOLOGIST: Dr. Sarah Smith
    """
    
    # Process the document text using our traditional extraction methods
    print("Step 1: Initial extraction with traditional methods")
    extracted_data = process_document_text(sample_report)
    
    print("\nExtracted data:")
    pprint(extracted_data)
    
    # Validate and enhance the extracted data
    print("\nStep 2: Validation and enhancement")
    validated_data = validate_report_data(extracted_data, sample_report)
    
    print("\nValidated data:")
    pprint(validated_data)
    
    # Test BIRADS extraction with different formats
    print("\n===== TESTING BIRADS EXTRACTION WITH DIFFERENT FORMATS =====\n")
    
    test_cases = [
        "BIRADS SCORE: 3",
        "BLRADS 2",
        "ACR Category 4a",
        "Assessment shows negative findings (BIRADS 1)",
        "Normal mammogram with no suspicious findings",
        "Findings are highly suspicious for malignancy",
        "Category 6 - Known biopsy-proven malignancy",
        "BIRADS classification - 0 (Additional imaging needed)"
    ]
    
    for test in test_cases:
        result = extract_birads_score(test)
        print(f"{test}\n  -> {result['value']} (confidence: {result['confidence']})\n")
    
    print("\n===== TESTING INCONSISTENT DATA SCENARIOS =====\n")
    
    # Test case with inconsistent BIRADS and impression
    inconsistent_report = """
    MAMMOGRAM REPORT
    
    EXAM DATE: 02/20/2023
    PATIENT: JANE SMITH
    
    FINDINGS:
    No suspicious masses or calcifications are seen.
    
    IMPRESSION:
    Normal screening mammogram with no evidence of malignancy.
    
    BIRADS SCORE: 4
    
    RECOMMENDATION:
    Routine screening mammogram in 1 year.
    """
    
    print("Processing report with inconsistent BIRADS and impression...")
    extracted_data = process_document_text(inconsistent_report)
    validated_data = validate_report_data(extracted_data, inconsistent_report)
    
    print("\nExtracted data:")
    pprint(extracted_data)
    
    print("\nValidated data (should fix inconsistency):")
    pprint(validated_data)

if __name__ == "__main__":
    run_tests() 