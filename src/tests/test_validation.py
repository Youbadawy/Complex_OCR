"""
Tests for the validation module.

This file contains tests for the validation functions 
used in the Medical Report Processor.
"""

import sys
import os
import unittest
from unittest.mock import patch, MagicMock

# Add the parent directory to the path so we can import the modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.document_processing.report_validation import (
    is_valid_date,
    is_valid_birads,
    is_valid_exam_type,
    normalize_date,
    normalize_birads,
    normalize_exam_type,
    check_birads_impression_consistency,
    extract_birads_from_impression,
    resolve_field_conflicts,
    fix_impression_duplication,
    augment_missing_fields,
    validate_report_data
)


class TestValidationFunctions(unittest.TestCase):
    """Test the basic validation functions."""

    def test_is_valid_date(self):
        """Test the is_valid_date function."""
        self.assertTrue(is_valid_date("2021-01-01"))
        self.assertFalse(is_valid_date(""))
        self.assertFalse(is_valid_date("Not Available"))
        self.assertFalse(is_valid_date("2021-13-01"))  # Invalid month
        self.assertFalse(is_valid_date("01/01/2021"))  # Wrong format

    def test_is_valid_birads(self):
        """Test the is_valid_birads function."""
        self.assertTrue(is_valid_birads("BIRADS 0"))
        self.assertTrue(is_valid_birads("BIRADS 4a"))
        self.assertTrue(is_valid_birads("birads 3"))
        self.assertFalse(is_valid_birads(""))
        self.assertFalse(is_valid_birads("Not Available"))
        self.assertFalse(is_valid_birads("Category 4"))  # Wrong format

    def test_is_valid_exam_type(self):
        """Test the is_valid_exam_type function."""
        self.assertTrue(is_valid_exam_type("MAMMOGRAM"))
        self.assertTrue(is_valid_exam_type("Breast Ultrasound"))
        self.assertFalse(is_valid_exam_type(""))
        self.assertFalse(is_valid_exam_type("Not Available"))
        self.assertFalse(is_valid_exam_type("X-Ray"))  # Not a valid type


class TestNormalizationFunctions(unittest.TestCase):
    """Test the normalization functions."""

    def test_normalize_date(self):
        """Test the normalize_date function."""
        self.assertEqual(normalize_date("2021-01-01"), "2021-01-01")
        self.assertEqual(normalize_date("01/01/2021"), "2021-01-01")
        self.assertEqual(normalize_date("January 1, 2021"), "2021-01-01")
        self.assertEqual(normalize_date("Jan 1 2021"), "2021-01-01")
        self.assertEqual(normalize_date("Not Available"), "Not Available")
        self.assertEqual(normalize_date(""), "")

    def test_normalize_birads(self):
        """Test the normalize_birads function."""
        self.assertEqual(normalize_birads("BIRADS 0"), "BIRADS 0")
        self.assertEqual(normalize_birads("birads4a"), "BIRADS 4a")
        self.assertEqual(normalize_birads("ACR 3"), "BIRADS 3")
        self.assertEqual(normalize_birads("Category 4"), "BIRADS 4")
        self.assertEqual(normalize_birads("Not Available"), "Not Available")

    def test_normalize_exam_type(self):
        """Test the normalize_exam_type function."""
        self.assertEqual(normalize_exam_type("MAMMOGRAM"), "MAMMOGRAM")
        self.assertEqual(normalize_exam_type("Bilateral Mammogram"), "MAMMOGRAM")
        self.assertEqual(normalize_exam_type("US"), "ULTRASOUND")
        self.assertEqual(normalize_exam_type("3D MAMMOGRAM"), "TOMOSYNTHESIS")
        self.assertEqual(normalize_exam_type("Not Available"), "Not Available")


class TestConsistencyFunctions(unittest.TestCase):
    """Test the consistency checking functions."""

    def test_check_birads_impression_consistency(self):
        """Test the check_birads_impression_consistency function."""
        # Consistent cases
        self.assertTrue(check_birads_impression_consistency(
            "BIRADS 1", "Normal examination with no evidence of malignancy."
        ))
        self.assertTrue(check_birads_impression_consistency(
            "BIRADS 2", "Stable benign finding, no suspicious features."
        ))
        self.assertTrue(check_birads_impression_consistency(
            "BIRADS 4", "Suspicious finding, recommend biopsy."
        ))
        
        # Inconsistent cases
        self.assertFalse(check_birads_impression_consistency(
            "BIRADS 1", "Suspicious finding, recommend biopsy."
        ))
        self.assertFalse(check_birads_impression_consistency(
            "BIRADS 4", "Normal examination with no evidence of malignancy."
        ))
        
        # Edge cases
        self.assertTrue(check_birads_impression_consistency(
            "", "Normal examination."
        ))
        self.assertTrue(check_birads_impression_consistency(
            "BIRADS 2", ""
        ))

    @patch('src.document_processing.report_validation.extract_birads_score')
    def test_extract_birads_from_impression(self, mock_extract):
        """Test the extract_birads_from_impression function."""
        # Test direct extraction
        mock_extract.return_value = {'value': 'BIRADS 4', 'confidence': 0.9}
        self.assertEqual(extract_birads_from_impression("Suspicious finding."), "BIRADS 4")
        
        # Test keyword-based extraction
        mock_extract.return_value = {'value': '', 'confidence': 0.0}
        self.assertEqual(
            extract_birads_from_impression("Normal examination with no evidence of malignancy."),
            "BIRADS 1"
        )
        self.assertEqual(
            extract_birads_from_impression("Stable benign finding."),
            "BIRADS 2"
        )
        self.assertEqual(
            extract_birads_from_impression("Suspicious finding, recommend biopsy."),
            "BIRADS 4"
        )
        
        # Test empty text
        self.assertEqual(extract_birads_from_impression(""), "")


class TestResolutionFunctions(unittest.TestCase):
    """Test the conflict resolution functions."""

    def test_resolve_field_conflicts(self):
        """Test the resolve_field_conflicts function."""
        # Test with confidence scores
        self.assertEqual(
            resolve_field_conflicts("test", ["Value1", "Value2"], [0.8, 0.9]),
            "Value2"
        )
        
        # Test BIRADS resolution
        self.assertEqual(
            resolve_field_conflicts("birads_score", ["BIRADS 2", "BIRADS 4"]),
            "BIRADS 4"  # Higher value is preferred
        )
        
        # Test date resolution
        self.assertEqual(
            resolve_field_conflicts("exam_date", ["2020-01-01", "2021-01-01"]),
            "2021-01-01"  # More recent date is preferred
        )
        
        # Test default resolution (longest value)
        self.assertEqual(
            resolve_field_conflicts("other_field", ["Short", "Longer value"]),
            "Longer value"
        )


class TestFixDuplicationFunction(unittest.TestCase):
    """Test the fix_impression_duplication function."""

    def test_fix_impression_duplication(self):
        """Test the fix_impression_duplication function."""
        # Test impression in findings
        report_data = {
            'impression': "Suspicious finding.",
            'findings': "Normal breast tissue. Suspicious finding."
        }
        fixed_data = fix_impression_duplication(report_data)
        self.assertEqual(fixed_data['findings'], "Normal breast tissue.")
        
        # Test recommendation in impression
        report_data = {
            'impression': "Suspicious finding. Recommend follow-up in 6 months.",
            'recommendation': "Recommend follow-up in 6 months."
        }
        fixed_data = fix_impression_duplication(report_data)
        self.assertEqual(fixed_data['impression'], "Suspicious finding.")
        
        # Test no duplication
        report_data = {
            'impression': "Suspicious finding.",
            'findings': "Normal breast tissue."
        }
        fixed_data = fix_impression_duplication(report_data)
        self.assertEqual(fixed_data, report_data)


class TestAugmentMissingFieldsFunction(unittest.TestCase):
    """Test the augment_missing_fields function."""

    @patch('src.document_processing.report_validation.extract_birads_from_impression')
    @patch('src.document_processing.report_validation.extract_birads_score')
    def test_augment_missing_fields(self, mock_extract_birads, mock_extract_from_impression):
        """Test the augment_missing_fields function."""
        # Setup mocks
        mock_extract_from_impression.return_value = "BIRADS 3"
        mock_extract_birads.return_value = {'value': 'BIRADS 4', 'confidence': 0.8}
        
        # Test extracting BIRADS from impression
        report_data = {
            'birads_score': "Not Available",
            'impression': "Probably benign finding."
        }
        augmented_data = augment_missing_fields(report_data, "Raw text")
        self.assertEqual(augmented_data['birads_score'], "BIRADS 3")
        
        # Test extracting BIRADS from raw text
        report_data = {
            'birads_score': "Not Available",
            'impression': ""
        }
        augmented_data = augment_missing_fields(report_data, "Raw text")
        self.assertEqual(augmented_data['birads_score'], "BIRADS 4")


class TestValidateReportDataFunction(unittest.TestCase):
    """Test the validate_report_data function."""

    @patch('src.document_processing.report_validation.normalize_date')
    @patch('src.document_processing.report_validation.normalize_birads')
    @patch('src.document_processing.report_validation.normalize_exam_type')
    @patch('src.document_processing.report_validation.fix_impression_duplication')
    @patch('src.document_processing.report_validation.check_birads_impression_consistency')
    @patch('src.document_processing.report_validation.extract_birads_from_impression')
    @patch('src.document_processing.report_validation.augment_missing_fields')
    def test_validate_report_data(
        self, mock_augment, mock_extract_birads, mock_check_consistency,
        mock_fix_duplication, mock_normalize_exam, mock_normalize_birads, mock_normalize_date
    ):
        """Test the validate_report_data function."""
        # Setup mocks
        mock_normalize_date.side_effect = lambda x: x
        mock_normalize_birads.side_effect = lambda x: x
        mock_normalize_exam.side_effect = lambda x: x
        mock_fix_duplication.side_effect = lambda x: x
        mock_check_consistency.return_value = True
        mock_extract_birads.return_value = ""
        mock_augment.side_effect = lambda x, y: x
        
        # Test basic validation flow
        report_data = {
            'exam_date': "2021-01-01",
            'birads_score': "BIRADS 2",
            'exam_type': "MAMMOGRAM",
            'impression': "Normal finding."
        }
        validated_data = validate_report_data(report_data, "Raw text")
        
        # Verify all normalization functions were called
        mock_normalize_date.assert_called_once_with("2021-01-01")
        mock_normalize_birads.assert_called_once_with("BIRADS 2")
        mock_normalize_exam.assert_called_once_with("MAMMOGRAM")
        
        # Verify fix_impression_duplication was called
        mock_fix_duplication.assert_called_once()
        
        # Verify consistency check was called
        mock_check_consistency.assert_called_once_with("BIRADS 2", "Normal finding.")
        
        # Verify augment_missing_fields was called
        mock_augment.assert_called_once()


if __name__ == '__main__':
    unittest.main() 