"""
Test module for text extraction functionality.
This module contains tests for the various extraction functions in the document_processing module.
"""

import unittest
import os
import sys
from pathlib import Path

# Add the parent directory to the path so we can import our modules
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.document_processing.text_analysis import (
    extract_birads_score,
    extract_provider_info,
    extract_signed_by,
    process_document_text
)

from src.document_processing.pdf_processor import (
    extract_sections,
    extract_patient_info,
    extract_date_from_text,
    extract_exam_type
)


class TestBiradsExtraction(unittest.TestCase):
    """Test cases for BIRADS score extraction"""
    
    def test_standard_birads_format(self):
        """Test standard BIRADS format extraction"""
        test_cases = [
            ("BIRADS SCORE: 4", "BIRADS 4", 0.95),
            ("BIRADS: 3", "BIRADS 3", 0.95),
            ("BI-RADS: 2", "BIRADS 2", 0.95),
            ("BIRADS ASSESSMENT: 0", "BIRADS 0", 0.95),
            ("BIRADS CLASSIFICATION: 5", "BIRADS 5", 0.95),
            ("BIRADS = 1", "BIRADS 1", 0.95),
        ]
        
        for text, expected_value, expected_confidence in test_cases:
            with self.subTest(text=text):
                result = extract_birads_score(text)
                self.assertEqual(result['value'], expected_value)
                self.assertEqual(result['confidence'], expected_confidence)
    
    def test_acr_format(self):
        """Test ACR format extraction"""
        test_cases = [
            ("ACR 4", "BIRADS 4", 0.85),
            ("Category 3", "BIRADS 3", 0.85),
            ("ACR SCORE: 2", "BIRADS 2", 0.85),
            ("Category ASSESSMENT: 0", "BIRADS 0", 0.85),
        ]
        
        for text, expected_value, expected_confidence in test_cases:
            with self.subTest(text=text):
                result = extract_birads_score(text)
                self.assertEqual(result['value'], expected_value)
                self.assertEqual(result['confidence'], expected_confidence)
    
    def test_assessment_format(self):
        """Test assessment format extraction"""
        test_cases = [
            ("Assessment: BIRADS 4", "BIRADS 4", 0.95),
            ("Impression: BI-RADS 3", "BIRADS 3", 0.95),
            ("Assessment: ACR 2", "BIRADS 2", 0.85),
        ]
        
        for text, expected_value, expected_confidence in test_cases:
            with self.subTest(text=text):
                result = extract_birads_score(text)
                self.assertEqual(result['value'], expected_value)
                self.assertEqual(result['confidence'], expected_confidence)
    
    def test_contextual_format(self):
        """Test contextual format extraction"""
        test_cases = [
            ("Mammogram shows Category 4a findings", "BIRADS 4a", 0.85),
            ("Findings consistent with BIRADS 0", "BIRADS 0", 0.95),
            ("Results indicating BIRADS 5", "BIRADS 5", 0.95),
        ]
        
        for text, expected_value, expected_confidence in test_cases:
            with self.subTest(text=text):
                result = extract_birads_score(text)
                self.assertEqual(result['value'], expected_value)
                self.assertEqual(result['confidence'], expected_confidence)
    
    def test_empty_text(self):
        """Test extraction with empty text"""
        result = extract_birads_score("")
        self.assertEqual(result['value'], "")
        self.assertEqual(result['confidence'], 0.0)
    
    def test_no_birads(self):
        """Test extraction with no BIRADS information"""
        result = extract_birads_score("No abnormalities detected in the mammogram.")
        self.assertEqual(result['value'], "")
        self.assertEqual(result['confidence'], 0.0)


class TestProviderExtraction(unittest.TestCase):
    """Test cases for provider information extraction"""
    
    def test_referring_provider(self):
        """Test referring provider extraction"""
        test_cases = [
            ("REFERRING PHYSICIAN: Dr. John Smith", "Dr. John Smith"),
            ("REFERRING: Dr. Jane Doe", "Dr. Jane Doe"),
            ("ORDERED BY: Dr. Robert Johnson", "Dr. Robert Johnson"),
            ("REFERRING PROVIDER: Dr. Emily Wilson, MD", "Dr. Emily Wilson, MD"),
        ]
        
        for text, expected in test_cases:
            with self.subTest(text=text):
                result = extract_provider_info(text)
                self.assertEqual(result['referring_provider'], expected)
    
    def test_interpreting_provider(self):
        """Test interpreting provider extraction"""
        test_cases = [
            ("INTERPRETING PHYSICIAN: Dr. John Smith", "Dr. John Smith"),
            ("INTERPRETED BY: Dr. Jane Doe", "Dr. Jane Doe"),
            ("READING PHYSICIAN: Dr. Robert Johnson", "Dr. Robert Johnson"),
            ("RADIOLOGIST: Dr. Emily Wilson, MD", "Dr. Emily Wilson, MD"),
        ]
        
        for text, expected in test_cases:
            with self.subTest(text=text):
                result = extract_provider_info(text)
                self.assertEqual(result['interpreting_provider'], expected)
    
    def test_facility(self):
        """Test facility extraction"""
        test_cases = [
            ("FACILITY: General Hospital", "General Hospital"),
            ("LOCATION: Medical Center", "Medical Center"),
            ("SITE: Community Clinic", "Community Clinic"),
            ("PERFORMED AT: University Hospital", "University Hospital"),
        ]
        
        for text, expected in test_cases:
            with self.subTest(text=text):
                result = extract_provider_info(text)
                self.assertEqual(result['facility'], expected)
    
    def test_empty_text(self):
        """Test extraction with empty text"""
        result = extract_provider_info("")
        self.assertEqual(result['referring_provider'], "")
        self.assertEqual(result['interpreting_provider'], "")
        self.assertEqual(result['facility'], "")


class TestSignatureExtraction(unittest.TestCase):
    """Test cases for signature extraction"""
    
    def test_signature_extraction(self):
        """Test signature extraction"""
        test_cases = [
            ("ELECTRONICALLY SIGNED BY: Dr. John Smith", "Dr. John Smith"),
            ("SIGNED BY: Dr. Jane Doe", "Dr. Jane Doe"),
            ("SIGNATURE: Dr. Robert Johnson", "Dr. Robert Johnson"),
            ("DICTATED BY: Dr. Emily Wilson, MD", "Dr. Emily Wilson, MD"),
            ("REPORTED BY: Dr. Michael Brown", "Dr. Michael Brown"),
        ]
        
        for text, expected in test_cases:
            with self.subTest(text=text):
                result = extract_signed_by(text)
                self.assertEqual(result, expected)
    
    def test_empty_text(self):
        """Test extraction with empty text"""
        result = extract_signed_by("")
        self.assertEqual(result, "")
    
    def test_no_signature(self):
        """Test extraction with no signature"""
        result = extract_signed_by("Report of mammogram findings")
        self.assertEqual(result, "")


class TestSectionExtraction(unittest.TestCase):
    """Test cases for section extraction"""
    
    def test_section_extraction(self):
        """Test section extraction from a complete report"""
        text = """
        CLINICAL HISTORY: 45-year-old female with family history of breast cancer.
        
        FINDINGS: Bilateral mammogram shows scattered fibroglandular densities. 
        No suspicious masses or microcalcifications identified.
        
        IMPRESSION: Negative mammogram, no evidence of malignancy.
        
        RECOMMENDATION: Routine screening mammogram in 1 year.
        """
        
        sections = extract_sections(text)
        
        self.assertIn("clinical_history", sections)
        self.assertIn("findings", sections)
        self.assertIn("impression", sections)
        self.assertIn("recommendation", sections)
        
        self.assertIn("45-year-old female with family history of breast cancer", sections["clinical_history"])
        self.assertIn("Bilateral mammogram shows scattered fibroglandular densities", sections["findings"])
        self.assertIn("Negative mammogram", sections["impression"])
        self.assertIn("Routine screening mammogram in 1 year", sections["recommendation"])
    
    def test_empty_text(self):
        """Test extraction with empty text"""
        sections = extract_sections("")
        self.assertEqual(sections, {})
    
    def test_missing_sections(self):
        """Test extraction with missing sections"""
        text = """
        CLINICAL HISTORY: 45-year-old female with family history of breast cancer.
        
        FINDINGS: Bilateral mammogram shows scattered fibroglandular densities.
        """
        
        sections = extract_sections(text)
        
        self.assertIn("clinical_history", sections)
        self.assertIn("findings", sections)
        self.assertNotIn("impression", sections)
        self.assertNotIn("recommendation", sections)


class TestPatientInfoExtraction(unittest.TestCase):
    """Test cases for patient information extraction"""
    
    def test_patient_info_extraction(self):
        """Test patient information extraction"""
        text = """
        PATIENT NAME: Jane Doe
        PATIENT ID: 12345
        PATIENT AGE: 45
        PATIENT GENDER: Female
        """
        
        info = extract_patient_info(text)
        
        self.assertIn("name", info)
        self.assertIn("id", info)
        self.assertIn("age", info)
        self.assertIn("gender", info)
        self.assertEqual(info["name"], "Jane Doe")
        self.assertEqual(info["id"], "12345")
        self.assertEqual(info["age"], "45")
        self.assertEqual(info["gender"], "Female")
    
    def test_empty_text(self):
        """Test extraction with empty text"""
        info = extract_patient_info("")
        self.assertEqual(info, {})


class TestDateExtraction(unittest.TestCase):
    """Test cases for date extraction"""
    
    def test_date_formats(self):
        """Test various date formats"""
        test_cases = [
            ("EXAM DATE: 2023-01-15", "2023-01-15"),
            ("DATE OF EXAM: 01/15/2023", "2023-01-15"),
            ("EXAMINATION DATE: Jan 15, 2023", "2023-01-15"),
        ]
        
        for text, expected in test_cases:
            with self.subTest(text=text):
                result = extract_date_from_text(text)
                self.assertEqual(result, expected)
    
    def test_empty_text(self):
        """Test extraction with empty text"""
        result = extract_date_from_text("")
        self.assertEqual(result, "")


class TestExamTypeExtraction(unittest.TestCase):
    """Test cases for exam type extraction"""
    
    def test_exam_types(self):
        """Test various exam types"""
        test_cases = [
            ("EXAM TYPE: MAMMOGRAM", "MAMMOGRAM"),
            ("EXAM: BREAST ULTRASOUND", "ULTRASOUND"),
            ("EXAMINATION: MRI BREAST", "MRI"),
        ]
        
        for text, expected in test_cases:
            with self.subTest(text=text):
                result = extract_exam_type(text)
                self.assertEqual(result, expected)
    
    def test_empty_text(self):
        """Test extraction with empty text"""
        result = extract_exam_type("")
        self.assertEqual(result, None)


class TestDocumentProcessing(unittest.TestCase):
    """Test cases for full document processing"""
    
    def test_full_document_processing(self):
        """Test processing a complete document"""
        # Create a test document with clean formatting to match processing expectations
        text = """PATIENT NAME: Jane Doe
PATIENT ID: 12345
PATIENT AGE: 45
PATIENT GENDER: Female

EXAM DATE: 2023-01-15
EXAM TYPE: MAMMOGRAM

CLINICAL HISTORY: 45-year-old female with family history of breast cancer.

FINDINGS: Bilateral mammogram shows scattered fibroglandular densities. 
No suspicious masses or microcalcifications identified.

IMPRESSION: Negative mammogram, BIRADS 1.

RECOMMENDATION: Routine screening mammogram in 1 year.

ELECTRONICALLY SIGNED BY: Dr. John Smith"""
        
        result = process_document_text(text)
        
        self.assertEqual(result['patient_name'], "Jane Doe")
        self.assertEqual(result['exam_date'], "2023-01-15")
        self.assertEqual(result['exam_type'], "MAMMOGRAM")
        self.assertEqual(result['birads_score'], "BIRADS 1")
        self.assertIn("family history of breast cancer", result['clinical_history'])
        self.assertIn("scattered fibroglandular densities", result['findings'])
        self.assertIn("Negative mammogram", result['impression'])
        self.assertIn("Routine screening mammogram in 1 year", result['recommendation'])
        self.assertEqual(result['signed_by'], "Dr. John Smith")
    
    def test_empty_text(self):
        """Test processing with empty text"""
        result = process_document_text("")
        self.assertEqual(result['patient_name'], "Not Available")
        self.assertEqual(result['exam_date'], "Not Available")
        self.assertEqual(result['exam_type'], "Not Available")
        self.assertEqual(result['birads_score'], "Not Available")


if __name__ == "__main__":
    unittest.main() 