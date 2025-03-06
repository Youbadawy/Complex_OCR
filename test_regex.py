#!/usr/bin/env python
"""
Test script for signature extraction regex patterns
"""
import re
import sys

def main():
    # Test text
    test_text = 'Signed by : Mrs Rosy Burns, CLERK, OT-PCS, 2020-12-09 Printed on: 2021-05-06 PROTEGE B Page: 1 Printed by: Sarah Holland'
    print(f"Test text: {test_text}")
    
    # Signature pattern with title capture
    pattern = r'Signed\s+by\s*:\s*(?P<title>Mrs|Miss|Ms|Mr|Dr)\.?\s+(?P<name>[^,\n]+)(?P<rest>(?:,\s*[^,\n]+){0,4})\s*,\s*(?P<date>\d{4}-\d{2}-\d{2}|\d{2}[/-]\d{2}[/-]\d{2,4})'
    
    # Test pattern
    match = re.search(pattern, test_text, re.IGNORECASE)
    if match:
        print(f"Match found: {match.group(0)}")
        print(f"Title: {match.group('title')}")
        print(f"Name: {match.group('name')}")
        print(f"Rest: {match.group('rest')}")
        print(f"Date: {match.group('date')}")
        
        # Clean up the signature
        title = match.group('title').strip()
        name = match.group('name').strip()
        rest = match.group('rest').strip()
        date = match.group('date').strip()
        
        # Normalize title
        if title.lower() in ['mrs', 'mr', 'dr', 'ms']:
            title = f"{title.title()}."
        
        # Fix title case for roles (CLERK → Clerk)
        def title_case_role(match):
            role = match.group(1)
            # Keep acronyms as-is (all caps with possible hyphens)
            if re.match(r'^[A-Z0-9\-]+$', role) and not role in ['CLERK', 'TECHNICIAN', 'DOCTOR', 'RADIOLOGIST', 'PHYSICIAN', 'ASSISTANT']:
                return role
            return role.title()
        
        rest = re.sub(r'\b([A-Z]{2,}[A-Za-z0-9\-]*)\b', title_case_role, rest)
        
        # Format the result
        result = f"{title} {name}{rest}, Signed on {date}"
        print(f"Cleaned result: {result}")
        
        # Expected output
        expected = "Mrs. Rosy Burns, Clerk, OT-PCS, Signed on 2020-12-09"
        print(f"Expected: {expected}")
        print(f"Match: {result == expected}")
    else:
        print("No match found")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 