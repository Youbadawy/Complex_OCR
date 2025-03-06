#!/usr/bin/env python
"""
Test script for provider extraction from medical documents
"""

import sys
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Import directly since we're in the src directory
from ocr_utils import extract_medical_fields

def main():
    # Example document text from the breast cancer screening report
    example_text = """
    PROTECTED B CFHS/SSFC  Defence mationale  
    Patient: ae  
    Scan-Mammo-Annual 
    Document Date: 2015-03-27  
    32684806  
    
    BC Cancer Agency SCREENING MAMMOGRAPHY 
    CARE & RESEARCH PROGRAM OF BC 
    An agency of the Provincial Health Services Authority  
    
    VICTCRIA SMP CENTRE  
    L378 7567  
    3C5 - 1990 FORT STREET 
    VICTORIA, BC V8R 6V4  
    TEL: (250) 952-4232  
    
    CL  
    *O101S2I*  
    
    SMPBC CENTRAL OFFICE (604) 877-6200  
    
    DR. CHRISTINA COBURN  
    CFS HEALTH SERVICES (PACIFIC) 
    97 - 1200 COLVILLE RD 
    VICTORIA BC V9A 7N2  
    
    Patient Name: 
    Date of Birth: 
    Date of Mammography: 27 MAR 15  
    
    INTERPRETATION: 
    - Bilateral Mammogram 
    Mammogram interpreted as within normal limits 
    
    RECOMMENDATION TO PRIMARY CARE PROVIDER:  
    If asymptomatic, return for regular screening in two years.  
    
    BRITISH COLUMBIA'S BREAST SCREENING POLICY 
    Average risk  
    Ages 40-49: Routine mammography available every two years. 
    Ages 50-74: Routine mammograms are recommended every two years.  
    
    ESPERO TRESS P  
    
    Higher than average risk  
    Ages 40-74 with a first degree relative with breast cancer: routine screening mammograms are recommended every year.  
    
    some cancers cannot be detected on a mammogram due to the location of the cancer or the density of the breast tissue. 
    Any new breast changes should be investigated as necessary.  
    
    For more information: www.screeningbe.ca
    """
    
    # Extract fields
    logging.info("Extracting fields from example text...")
    results = extract_medical_fields(example_text)
    
    # Print results in pretty format
    print("\nExtraction Results:")
    print(json.dumps(results, indent=2))
    
    # Specifically highlight provider information
    print("\nProvider Information:")
    print(f"Facility Name: {results.get('facility_name', 'Not extracted')}")
    print(f"Referring Provider: {results.get('referring_provider', 'Not extracted')}")
    print(f"Testing Provider: {results.get('testing_provider', 'Not extracted')}")
    
    # Check if redacted
    print(f"\nRedacted Document: {results.get('is_redacted', False)}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 