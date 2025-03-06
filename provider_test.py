#!/usr/bin/env python
"""
Standalone test script for provider extraction from medical documents
"""

import re
import json

def extract_provider_info(text):
    """
    Extract provider information from text with enhanced pattern matching for government/military health services
    """
    if not text:
        return {}
    
    results = {}
    
    # Enhanced patterns for radiologist/reporting doctor
    radiologist_patterns = [
        # High confidence patterns
        (r'(?:signed|electronically signed|dictated)(?:\s+by)?[\s:]+(?:DR\.?|Doctor|Dr\.?)\s+([A-Z][A-Za-z\'\-]+(?:\s+[A-Z][A-Za-z\'\-]+){1,3})', 0.95),
        (r'(?:radiologist|reporting physician|physician|provider|doctor)[\s:]+(?:DR\.?|Doctor|Dr\.?)\s+([A-Z][A-Za-z\'\-]+(?:\s+[A-Z][A-Za-z\'\-]+){1,3})', 0.95),
        (r'(?:DR\.?|Doctor|Dr\.?)\s+([A-Z][A-Za-z\'\-]+(?:\s+[A-Z][A-Za-z\'\-]+){1,3})(?:,\s+(?:MD|M\.D\.|R\.?T\.?|radiologist|physician))', 0.95),
        
        # Medium confidence patterns
        (r'(?:DR\.?|Doctor|Dr\.?)\s+([A-Z][A-Za-z\'\-]+(?:\s+[A-Z][A-Za-z\'\-]+){1,3})', 0.85),
        (r'(?:signed|electronically signed|dictated)(?:\s+by)?[\s:]+([A-Z][A-Za-z\'\-]+(?:\s+[A-Z][A-Za-z\'\-]+){1,3})(?:,\s+(?:MD|M\.D\.|R\.?T\.?|radiologist|physician))', 0.8),
        
        # Lower confidence patterns
        (r'(?:reported by|read by|interpreted by)[\s:]+([A-Z][A-Za-z\'\-]+(?:\s+[A-Z][A-Za-z\'\-]+){1,3})', 0.7),
    ]
    
    # Enhanced patterns for referring doctor
    referring_patterns = [
        # High confidence patterns
        (r'(?:referring|ordering)(?:\s+physician|\s+doctor|\s+provider)?[\s:]+(?:DR\.?|Doctor|Dr\.?)\s+([A-Z][A-Za-z\'\-]+(?:\s+[A-Z][A-Za-z\'\-]+){1,3})', 0.95),
        (r'(?:referr(?:ed|ing)\s+(?:by|from)|ordered\s+by|request(?:ed)?\s+(?:by|from))[\s:]+(?:DR\.?|Doctor|Dr\.?)\s+([A-Z][A-Za-z\'\-]+(?:\s+[A-Z][A-Za-z\'\-]+){1,3})', 0.9),
        
        # Military/government specific patterns (high confidence)
        (r'(?:CFS|CF|DND|CFHS|SSFC)\s+HEALTH\s+SERVICES[\s\.\,]*(?:\([A-Z]+\))?[\r\n\s]*(?:DR\.?|Doctor|Dr\.?)\s+([A-Z][A-Za-z\'\-]+(?:\s+[A-Z][A-Za-z\'\-]+){1,3})', 0.95),
        
        # Medium confidence patterns
        (r'(?:ordering\s+provider|ordering\s+physician|referring\s+physician)[\s:]+([A-Z][A-Za-z\'\-]+(?:\s+[A-Z][A-Za-z\'\-]+){1,3})', 0.8),
        (r'(?:referr(?:ed|ing)|ordered|request(?:ed)?)(?:\s+by|\s+from)?[\s:]+([A-Z][A-Za-z\'\-]+(?:\s+[A-Z][A-Za-z\'\-]+){1,3})', 0.7),
        
        # Government facility patterns (separate line)
        (r'(?:CFS|CF|DND|CFHS|SSFC)\s+HEALTH\s+SERVICES(?:\s+\([A-Z]+\))?', 0.9)
    ]
    
    # Facility patterns for medical centers
    facility_patterns = [
        # High confidence patterns
        (r'((?:[A-Z][a-z]+\s+)?(?:Cancer|Medical|Health|Imaging|Radiology|Hospital|Clinic)(?:\s+[A-Za-z]+){1,6})\s+(?:CENTER|CENTRE|PROGRAM|AGENCY)', 0.9),
        (r'((?:[A-Z][A-Za-z]+\s+){1,3}(?:CLINIC|HOSPITAL|CENTER|CENTRE|AGENCY))', 0.9),
        (r'((?:[A-Z][A-Za-z]+\s+){1,3}(?:MEDICAL|HEALTH|IMAGING|RADIOLOGY)(?:\s+[A-Za-z]+){1,3})', 0.85),
        
        # BC Cancer Agency specific
        (r'(BC\s+Cancer\s+Agency[\s\w\&]+(?:PROGRAM|CENTRE|CENTER)[\s\w\&]+)', 0.95),
        
        # Military specific
        (r'((?:CFS|CF|DND|CFHS|SSFC)\s+HEALTH\s+SERVICES(?:\s+\([A-Z]+\))?(?:[\r\n\s]+[\w\-\s\,\.]+){0,3})', 0.95)
    ]
    
    # Process facility names first to establish context
    facility_candidates = []
    facility_found = False
    
    for pattern, confidence in facility_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            facility_name = match.group(1).strip()
            if facility_name and len(facility_name) > 5:  # Minimum length check
                facility_found = True
                # Clean up facility name - remove newlines and extra spaces
                facility_name = re.sub(r'\s+', ' ', facility_name.replace('\n', ' '))
                facility_candidates.append({
                    'value': facility_name,
                    'confidence': confidence
                })
    
    # If facility was found, add to results
    if facility_candidates:
        # Sort by confidence and pick the best
        facility_candidates.sort(key=lambda x: x['confidence'], reverse=True)
        results['facility_name'] = facility_candidates[0]
    
    # Extract radiologist information
    radiologist_candidates = []
    
    for pattern, confidence in radiologist_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            provider_name = match.group(1).strip()
            if provider_name and not _is_invalid_signature(provider_name):
                # Clean up provider name - remove newlines and extra spaces
                provider_name = re.sub(r'\s+', ' ', provider_name.replace('\n', ' '))
                # Limit to first two words (first and last name)
                name_parts = provider_name.split()
                if len(name_parts) > 2:
                    provider_name = ' '.join(name_parts[:2])
                radiologist_candidates.append({
                    'value': provider_name,
                    'confidence': confidence
                })
    
    # Extract referring provider information
    referring_candidates = []
    referring_facility = None
    
    for pattern, confidence in referring_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            # Check if this is a facility pattern (group 1 might not exist)
            if len(match.groups()) == 0:
                referring_facility = match.group(0).strip()
                continue
                
            provider_name = match.group(1).strip()
            if provider_name and not _is_invalid_signature(provider_name):
                # Clean up provider name - remove newlines and extra spaces
                provider_name = re.sub(r'\s+', ' ', provider_name.replace('\n', ' '))
                # Limit to first two words (first and last name)
                name_parts = provider_name.split()
                if len(name_parts) > 2:
                    provider_name = ' '.join(name_parts[:2])
                referring_candidates.append({
                    'value': provider_name,
                    'confidence': confidence
                })
    
    # Look for military/government specific formats - these often have provider info on separate lines
    if not referring_candidates and ('CFS HEALTH SERVICES' in text or 'CFHS' in text or 'SSFC' in text):
        # Check for DR name in close proximity to CFS HEALTH SERVICES
        military_provider_pattern = r'DR\.?\s+([A-Z][A-Za-z\'\-]+(?:\s+[A-Z][A-Za-z\'\-]+){1,3})'
        matches = re.finditer(military_provider_pattern, text, re.IGNORECASE)
        for match in matches:
            provider_name = match.group(1).strip()
            if provider_name and not _is_invalid_signature(provider_name):
                # Clean up provider name - remove newlines and extra spaces
                provider_name = re.sub(r'\s+', ' ', provider_name.replace('\n', ' '))
                # Limit to first two words (first and last name)
                name_parts = provider_name.split()
                if len(name_parts) > 2:
                    provider_name = ' '.join(name_parts[:2])
                
                pos = match.start()
                context_start = max(0, pos - 300)
                context_end = min(len(text), pos + 300)
                context = text[context_start:context_end]
                
                if 'CFS HEALTH SERVICES' in context or 'CFHS' in context or 'SSFC' in context:
                    referring_candidates.append({
                        'value': provider_name,
                        'confidence': 0.85
                    })
    
    # Sort candidates by confidence
    if radiologist_candidates:
        radiologist_candidates.sort(key=lambda x: x['confidence'], reverse=True)
        results['testing_provider'] = radiologist_candidates[0]
    
    if referring_candidates:
        referring_candidates.sort(key=lambda x: x['confidence'], reverse=True)
        results['referring_provider'] = referring_candidates[0]
    
    # Clean up extracted names to remove titles
    for key in results:
        if isinstance(results[key], dict) and 'value' in results[key]:
            # Remove titles and qualifications from names
            name = results[key]['value']
            name = re.sub(r'^(?:Dr\.?|Doctor|DOCTOR)\s+', '', name, flags=re.IGNORECASE)
            name = re.sub(r',\s*(?:MD|M\.D\.|Ph\.?D\.?|R\.?T\.?|radiologist|physician).*$', '', name, flags=re.IGNORECASE)
            results[key]['value'] = name
    
    # Add CFS HEALTH SERVICES as facility if not already found
    if 'facility_name' not in results and 'CFS HEALTH SERVICES' in text:
        results['facility_name'] = {
            'value': 'CFS HEALTH SERVICES',
            'confidence': 0.9
        }
    
    # Add BC Cancer Agency as facility if not already found
    if 'facility_name' not in results and 'BC Cancer Agency' in text:
        results['facility_name'] = {
            'value': 'BC Cancer Agency SCREENING MAMMOGRAPHY',
            'confidence': 0.9
        }
    
    return results

def _is_invalid_signature(text):
    """Check if a potential signature is invalid"""
    if not text or len(text) < 3:
        return True
        
    # Check for common non-name terms
    invalid_terms = [
        'report', 'dictated', 'transcribed', 'signed', 'electronically', 
        'date', 'time', 'page', 'fax', 'phone', 'tel', 'signature',
        'please', 'thank', 'regards', 'sincerely', 'attachment'
    ]
    
    text_lower = text.lower()
    for term in invalid_terms:
        if term in text_lower:
            return True
            
    # Check for non-name patterns
    if re.search(r'\d{1,2}:\d{2}', text):  # Time format
        return True
    if re.search(r'\d{2}/\d{2}/\d{2,4}', text):  # Date format
        return True
    if re.search(r'[<>{}[\]#%&@$*]', text):  # Special characters
        return True
        
    return False

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
    
    # Extract provider information
    print("Extracting provider information from example text...")
    results = extract_provider_info(example_text)
    
    # Print results in pretty format
    print("\nExtraction Results:")
    print(json.dumps(results, indent=2))
    
    # Specifically highlight provider information
    print("\nProvider Information:")
    if 'facility_name' in results:
        print(f"Facility Name: {results['facility_name']['value']} (confidence: {results['facility_name']['confidence']})")
    else:
        print("Facility Name: Not extracted")
        
    if 'referring_provider' in results:
        print(f"Referring Provider: {results['referring_provider']['value']} (confidence: {results['referring_provider']['confidence']})")
    else:
        print("Referring Provider: Not extracted")
        
    if 'testing_provider' in results:
        print(f"Testing Provider: {results['testing_provider']['value']} (confidence: {results['testing_provider']['confidence']})")
    else:
        print("Testing Provider: Not extracted")
    
    return 0

if __name__ == "__main__":
    main() 