#!/usr/bin/env python
"""
Test script for signature block extraction
"""
import sys
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    try:
        # Test importing the module
        logging.info("Importing module...")
        from src.document_processing.text_analysis import extract_signature_block, clean_signature_block
        logging.info("Import successful")
        
        # Test example
        test_text = 'Signed by : Mrs Rosy Burns, CLERK, OT-PCS, 2020-12-09 Printed on: 2021-05-06 PROTEGE B Page: 1 Printed by: Sarah Holland'
        logging.info(f"Test text: {test_text}")
        
        # Test direct extraction
        result = extract_signature_block(test_text)
        logging.info(f"Extract result: {result}")
        
        # Test just cleaning
        cleaned = clean_signature_block('Mrs Rosy Burns, CLERK, OT-PCS, 2020-12-09 Printed on: 2021-05-06')
        logging.info(f"Clean result: {cleaned}")
        
        # Expected output: "Mrs. Rosy Burns, Clerk, OT-PCS, Signed on 2020-12-09"
        logging.info(f"Is extraction successful? {result.get('value') == 'Mrs. Rosy Burns, Clerk, OT-PCS, Signed on 2020-12-09'}")
        
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 