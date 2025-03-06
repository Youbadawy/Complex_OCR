"""
Validation pipeline for medical report extraction.

This module provides functionality for validating and cross-checking extracted 
information from medical reports, enabling the resolution of conflicts between 
different extraction methods and ensuring consistency in the final data.
"""

import re
import logging
from typing import Dict, Any, List, Optional, Tuple
import datetime
from dataclasses import dataclass

# Set up logger
logger = logging.getLogger(__name__)

@dataclass
class ValidationRule:
    """A rule for validating extracted fields"""
    field_name: str
    validation_function: callable
    description: str
    severity: str = "warning"  # warning, error, info
    
    def __post_init__(self):
        # Validate severity level
        if self.severity not in ["warning", "error", "info"]:
            raise ValueError(f"Invalid severity level: {self.severity}")

class ValidationResult:
    """Results from validation rules"""
    
    def __init__(self):
        self.issues = []
        self.modifications = {}
        
    def add_issue(self, rule: ValidationRule, message: str, field_name: str, 
                 original_value: Any, suggested_value: Any = None):
        """
        Add a validation issue
        
        Args:
            rule: The validation rule that was violated
            message: Description of the issue
            field_name: Name of the field with the issue
            original_value: Original value of the field
            suggested_value: Suggested correction (if any)
        """
        self.issues.append({
            "rule_description": rule.description,
            "severity": rule.severity,
            "message": message,
            "field": field_name,
            "original_value": original_value,
            "suggested_value": suggested_value
        })
        
        # Store modification if a suggested value is provided
        if suggested_value is not None:
            self.modifications[field_name] = suggested_value
    
    def has_errors(self) -> bool:
        """Check if there are any error-level issues"""
        return any(issue["severity"] == "error" for issue in self.issues)
    
    def has_warnings(self) -> bool:
        """Check if there are any warning-level issues"""
        return any(issue["severity"] == "warning" for issue in self.issues)
    
    def get_modified_data(self, original_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply suggested modifications to the original data
        
        Args:
            original_data: Original extracted data
            
        Returns:
            Modified data with corrections applied
        """
        modified_data = original_data.copy()
        for field, value in self.modifications.items():
            if field in modified_data:
                if isinstance(modified_data[field], dict) and "value" in modified_data[field]:
                    # Handle fields stored as {value: X, confidence: Y}
                    modified_data[field]["value"] = value
                    # Reduce confidence slightly since this is a correction
                    if "confidence" in modified_data[field]:
                        modified_data[field]["confidence"] *= 0.9
                else:
                    # Handle direct field values
                    modified_data[field] = value
            else:
                # Add new field
                modified_data[field] = value
        
        return modified_data

class ReportValidator:
    """
    Validator for medical report extracted data.
    
    This class provides methods for validating extracted medical report data,
    including cross-checking between different extraction methods, verifying 
    format consistency, and ensuring field values are correct and consistent.
    """
    
    def __init__(self):
        """Initialize the validator with standard validation rules"""
        self.rules = self._create_validation_rules()
    
    def _create_validation_rules(self) -> List[ValidationRule]:
        """
        Create the standard set of validation rules
        
        Returns:
            List of validation rules for medical reports
        """
        rules = []
        
        # BIRADS score validation
        rules.append(ValidationRule(
            field_name="birads_score",
            validation_function=self._validate_birads_score,
            description="BIRADS score should be in a valid format (BIRADS 0-6)",
            severity="error"
        ))
        
        # Date format validation
        rules.append(ValidationRule(
            field_name="exam_date",
            validation_function=self._validate_date_format,
            description="Date should be in YYYY-MM-DD format",
            severity="warning"
        ))
        
        # Provider name validation
        rules.append(ValidationRule(
            field_name="provider_name",
            validation_function=self._validate_provider_name,
            description="Provider name should have proper capitalization and spacing",
            severity="info"
        ))
        
        # Duplicate field detection (impression/findings)
        rules.append(ValidationRule(
            field_name="impression",
            validation_function=self._validate_impression_vs_findings,
            description="Impression should not be an exact duplicate of findings",
            severity="warning"
        ))
        
        # Content consistency validation
        rules.append(ValidationRule(
            field_name="content_consistency",
            validation_function=self._validate_content_consistency,
            description="Fields should be consistent with each other",
            severity="warning"
        ))
        
        return rules
    
    def _validate_birads_score(self, data: Dict[str, Any]) -> Tuple[bool, str, Any]:
        """
        Validate BIRADS score format and value
        
        Args:
            data: Extracted report data
            
        Returns:
            Tuple of (is_valid, message, suggested_correction)
        """
        if "birads_score" not in data:
            return True, "", None
            
        birads_value = data["birads_score"].get("value", "") if isinstance(data["birads_score"], dict) else data["birads_score"]
        
        # Skip empty values
        if not birads_value:
            return True, "", None
            
        # Check format
        if not re.match(r'BIRADS\s*[0-6][abc]?', birads_value, re.IGNORECASE):
            # Try to extract a valid format from what we have
            match = re.search(r'[0-6][abc]?', birads_value)
            if match:
                suggested = f"BIRADS {match.group(0)}"
                return False, f"Invalid BIRADS format: '{birads_value}'", suggested
            else:
                return False, f"Invalid BIRADS format: '{birads_value}'", None
                
        # Normalize to standard format
        normalized = re.sub(r'BIRADS\s+', 'BIRADS ', birads_value, flags=re.IGNORECASE)
        if normalized.upper() != birads_value.upper():
            return False, f"Non-standard BIRADS format: '{birads_value}'", normalized
            
        return True, "", None
    
    def _validate_date_format(self, data: Dict[str, Any]) -> Tuple[bool, str, Any]:
        """
        Validate date format
        
        Args:
            data: Extracted report data
            
        Returns:
            Tuple of (is_valid, message, suggested_correction)
        """
        if "exam_date" not in data:
            return True, "", None
            
        date_value = data["exam_date"].get("value", "") if isinstance(data["exam_date"], dict) else data["exam_date"]
        
        # Skip empty values
        if not date_value:
            return True, "", None
            
        # Try to parse different date formats and normalize
        try:
            # Check if already in ISO format
            if re.match(r'\d{4}-\d{2}-\d{2}', date_value):
                # Verify it's a valid date
                year, month, day = map(int, date_value.split('-'))
                datetime.date(year, month, day)  # Will raise ValueError if invalid
                return True, "", None
                
            # Try MM/DD/YYYY format
            match = re.match(r'(\d{1,2})/(\d{1,2})/(\d{4})', date_value)
            if match:
                month, day, year = map(int, match.groups())
                # Create date to validate
                date_obj = datetime.date(year, month, day)
                return False, f"Date not in ISO format: '{date_value}'", date_obj.isoformat()
                
            # Try DD-MM-YYYY format
            match = re.match(r'(\d{1,2})-(\d{1,2})-(\d{4})', date_value)
            if match:
                day, month, year = map(int, match.groups())
                # Create date to validate
                date_obj = datetime.date(year, month, day)
                return False, f"Date not in ISO format: '{date_value}'", date_obj.isoformat()
                
            # Try Month DD, YYYY format
            match = re.match(r'([A-Za-z]+)\s+(\d{1,2}),?\s*(\d{4})', date_value)
            if match:
                month_name, day, year = match.groups()
                month_mapping = {
                    'january': 1, 'february': 2, 'march': 3, 'april': 4, 'may': 5, 'june': 6,
                    'july': 7, 'august': 8, 'september': 9, 'october': 10, 'november': 11, 'december': 12
                }
                month = month_mapping.get(month_name.lower())
                if month:
                    date_obj = datetime.date(int(year), month, int(day))
                    return False, f"Date not in ISO format: '{date_value}'", date_obj.isoformat()
                    
            # Cannot parse the date
            return False, f"Invalid date format: '{date_value}'", None
            
        except ValueError as e:
            return False, f"Invalid date: '{date_value}' - {str(e)}", None
    
    def _validate_provider_name(self, data: Dict[str, Any]) -> Tuple[bool, str, Any]:
        """
        Validate provider name format
        
        Args:
            data: Extracted report data
            
        Returns:
            Tuple of (is_valid, message, suggested_correction)
        """
        if "provider_name" not in data:
            return True, "", None
            
        provider_value = data["provider_name"].get("value", "") if isinstance(data["provider_name"], dict) else data["provider_name"]
        
        # Skip empty values
        if not provider_value:
            return True, "", None
            
        # Check for all uppercase or all lowercase
        if provider_value.isupper() or provider_value.islower():
            # Properly capitalize
            words = provider_value.split()
            capitalized = []
            for word in words:
                if word.lower() in ['md', 'do', 'pa', 'np', 'rn', 'phd']:
                    capitalized.append(word.upper())
                elif word.lower() in ['van', 'de', 'la', 'von', 'del']:
                    capitalized.append(word.lower())
                else:
                    capitalized.append(word.capitalize())
            
            suggested = ' '.join(capitalized)
            return False, f"Provider name formatting issues: '{provider_value}'", suggested
            
        return True, "", None
    
    def _validate_impression_vs_findings(self, data: Dict[str, Any]) -> Tuple[bool, str, Any]:
        """
        Validate impression field isn't duplicate of findings
        
        Args:
            data: Extracted report data
            
        Returns:
            Tuple of (is_valid, message, suggested_correction)
        """
        if "impression" not in data or "findings" not in data:
            return True, "", None
            
        impression = data["impression"].get("value", "") if isinstance(data["impression"], dict) else data["impression"]
        findings = data["findings"].get("value", "") if isinstance(data["findings"], dict) else data["findings"]
        
        # Skip empty values
        if not impression or not findings:
            return True, "", None
            
        # Check for duplicate content
        if impression.strip() == findings.strip():
            return False, "Impression is an exact duplicate of findings", None
            
        # Check for near-duplicate (>90% match)
        impression_words = set(impression.lower().split())
        findings_words = set(findings.lower().split())
        
        if impression_words and findings_words:
            overlap = len(impression_words.intersection(findings_words))
            similarity = overlap / min(len(impression_words), len(findings_words))
            
            if similarity > 0.9:
                return False, f"Impression is very similar to findings (similarity: {similarity:.2f})", None
                
        return True, "", None
    
    def _validate_content_consistency(self, data: Dict[str, Any]) -> Tuple[bool, str, Any]:
        """
        Validate consistency between fields
        
        Args:
            data: Extracted report data
            
        Returns:
            Tuple of (is_valid, message, suggested_correction)
        """
        issues = []
        
        # Check BIRADS consistency with impression/findings
        birads_value = ""
        if "birads_score" in data:
            birads_value = data["birads_score"].get("value", "") if isinstance(data["birads_score"], dict) else data["birads_score"]
        
        impression = ""
        if "impression" in data:
            impression = data["impression"].get("value", "") if isinstance(data["impression"], dict) else data["impression"]
            
        findings = ""
        if "findings" in data:
            findings = data["findings"].get("value", "") if isinstance(data["findings"], dict) else data["findings"]
            
        # If we have both BIRADS and impression/findings
        if birads_value and (impression or findings):
            birads_match = re.search(r'BIRADS\s*([0-6][abc]?)', birads_value, re.IGNORECASE)
            if birads_match:
                birads_number = birads_match.group(1)
                
                # Check if the birads number appears in impression or findings
                combined_text = (impression + " " + findings).lower()
                
                # Look for variations like "category 4" or "bi-rads 4" in impression/findings
                alt_patterns = [
                    rf'bi-?rads\s*{birads_number}',
                    rf'category\s*{birads_number}',
                    rf'acr\s*{birads_number}'
                ]
                
                matches_found = False
                for pattern in alt_patterns:
                    if re.search(pattern, combined_text, re.IGNORECASE):
                        matches_found = True
                        break
                
                # If BIRADS isn't mentioned in impression/findings when it should be
                if not matches_found and birads_number not in ['0', '1']:
                    issues.append(f"BIRADS score {birads_number} is not reflected in impression or findings")
        
        if issues:
            return False, ", ".join(issues), None
        
        return True, "", None
    
    def validate(self, extracted_data: Dict[str, Any]) -> ValidationResult:
        """
        Validate extracted data against all rules
        
        Args:
            extracted_data: The data extracted from a medical report
            
        Returns:
            ValidationResult object with issues and suggested corrections
        """
        result = ValidationResult()
        
        # Apply each validation rule
        for rule in self.rules:
            try:
                # Some rules operate on the entire data structure
                if rule.field_name == "content_consistency":
                    is_valid, message, suggested = rule.validation_function(extracted_data)
                    if not is_valid:
                        result.add_issue(rule, message, "multiple_fields", "N/A", suggested)
                    continue
                
                # Skip rules for fields that don't exist in the data
                if rule.field_name not in extracted_data:
                    continue
                
                # Apply the validation rule
                is_valid, message, suggested = rule.validation_function(extracted_data)
                if not is_valid:
                    field_value = extracted_data[rule.field_name]
                    if isinstance(field_value, dict) and "value" in field_value:
                        original_value = field_value["value"]
                    else:
                        original_value = field_value
                        
                    result.add_issue(rule, message, rule.field_name, original_value, suggested)
            
            except Exception as e:
                logger.error(f"Error applying validation rule {rule.description}: {str(e)}")
                
        return result

def validate_and_enhance_extraction(extraction_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and enhance extraction results, applying corrections where needed.
    
    Args:
        extraction_results: Data extracted from medical report text
        
    Returns:
        Enhanced and validated extraction results
    """
    # Create validator
    validator = ReportValidator()
    
    # Run validation
    validation_result = validator.validate(extraction_results)
    
    # Apply suggested corrections
    if validation_result.modifications:
        logger.info(f"Applying {len(validation_result.modifications)} corrections to extracted data")
        enhanced_data = validation_result.get_modified_data(extraction_results)
        
        # Log validation issues
        for issue in validation_result.issues:
            if issue["severity"] == "error":
                logger.error(f"Validation error: {issue['message']}")
            elif issue["severity"] == "warning":
                logger.warning(f"Validation warning: {issue['message']}")
            else:
                logger.info(f"Validation info: {issue['message']}")
                
        return enhanced_data
    
    # No modifications needed
    return extraction_results

def cross_validate_extractions(rule_based_results: Dict[str, Any], 
                              llm_based_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Cross-validate and merge results from different extraction methods.
    
    Args:
        rule_based_results: Results from rule-based extraction
        llm_based_results: Results from LLM-based extraction
        
    Returns:
        Merged and cross-validated results
    """
    if not rule_based_results and not llm_based_results:
        logger.warning("Both extraction results are empty")
        return {}
        
    if not rule_based_results:
        logger.info("Only LLM-based results available")
        return validate_and_enhance_extraction(llm_based_results)
        
    if not llm_based_results:
        logger.info("Only rule-based results available")
        return validate_and_enhance_extraction(rule_based_results)
    
    # Start with rule-based results as baseline
    merged_results = rule_based_results.copy()
    
    # Track conflicts for logging
    conflicts = []
    
    # Merge fields from LLM extraction, prioritizing based on confidence
    for field, llm_value in llm_based_results.items():
        # Skip if LLM didn't extract this field
        if not llm_value:
            continue
            
        # Different handling based on whether the field exists in rule-based results
        if field not in merged_results or not merged_results[field]:
            # Field missing in rule-based results, use LLM result
            logger.debug(f"Using LLM extraction for missing field: {field}")
            merged_results[field] = llm_value
        else:
            # Both extractions have this field, compare and choose best
            rule_value = merged_results[field]
            
            # Handle different data structures
            rule_confidence = 0.0
            llm_confidence = 0.0
            
            if isinstance(rule_value, dict) and "confidence" in rule_value:
                rule_confidence = rule_value["confidence"]
                rule_text_value = rule_value.get("value", "")
            else:
                rule_text_value = rule_value
                
            if isinstance(llm_value, dict) and "confidence" in llm_value:
                llm_confidence = llm_value["confidence"]
                llm_text_value = llm_value.get("value", "")
            else:
                llm_text_value = llm_value
            
            # Compare values
            if rule_text_value == llm_text_value:
                # Values match, increase confidence
                if isinstance(merged_results[field], dict) and "confidence" in merged_results[field]:
                    merged_results[field]["confidence"] = min(0.99, merged_results[field]["confidence"] + 0.1)
            else:
                # Values don't match, use the one with higher confidence
                if llm_confidence > rule_confidence:
                    logger.debug(f"Using LLM extraction for field {field} due to higher confidence")
                    merged_results[field] = llm_value
                    conflicts.append(f"Field '{field}': Rule='{rule_text_value}', LLM='{llm_text_value}' (chose LLM)")
                else:
                    conflicts.append(f"Field '{field}': Rule='{rule_text_value}', LLM='{llm_text_value}' (kept Rule)")
    
    # Log conflicts for debugging
    if conflicts:
        logger.info(f"Resolved {len(conflicts)} conflicts between extraction methods")
        for conflict in conflicts:
            logger.debug(f"Conflict resolution: {conflict}")
    
    # Validate the merged results
    return validate_and_enhance_extraction(merged_results)

def validate_report_data(extracted_data: Dict[str, Any], raw_text: str = None) -> Dict[str, Any]:
    """
    Validate and enhance extracted report data.
    
    Args:
        extracted_data: Dictionary of extracted fields
        raw_text: Optional raw text for additional validation
        
    Returns:
        Dictionary with validated and enhanced data
    """
    logger.info("Validating extracted report data")
    
    # Create a copy to avoid modifying the original
    validated_data = extracted_data.copy()
    
    # Apply validation rules
    validation_result = apply_validation_rules(validated_data, raw_text)
    
    # Apply suggested modifications from validation
    for field, value in validation_result.modifications.items():
        validated_data[field] = value
        logger.debug(f"Modified field '{field}' based on validation")
    
    # Log validation issues
    for issue in validation_result.issues:
        if issue['rule'].severity == 'error':
            logger.error(f"Validation error in {issue['field_name']}: {issue['message']}")
        elif issue['rule'].severity == 'warning':
            logger.warning(f"Validation warning in {issue['field_name']}: {issue['message']}")
        else:
            logger.info(f"Validation note for {issue['field_name']}: {issue['message']}")
    
    # Enhance data with additional processing
    validated_data = enhance_extracted_data(validated_data, raw_text)
    
    return validated_data

def extract_and_validate_report(raw_text: str) -> Dict[str, Any]:
    """
    Extract and validate information from a medical report.
    
    This function combines extraction and validation in a single step.
    
    Args:
        raw_text: Raw text from the medical report
        
    Returns:
        Dictionary with extracted and validated data
    """
    logger.info("Extracting and validating report data")
    
    # Import here to avoid circular imports
    from .text_analysis import extract_structured_data
    
    # Extract structured data
    extracted_data = extract_structured_data(raw_text)
    
    # Validate and enhance the extracted data
    validated_data = validate_report_data(extracted_data, raw_text)
    
    return validated_data

def apply_validation_rules(data: Dict[str, Any], raw_text: str = None) -> ValidationResult:
    """
    Apply validation rules to extracted data.
    
    Args:
        data: Dictionary of extracted fields
        raw_text: Optional raw text for additional validation
        
    Returns:
        ValidationResult object with issues and suggested modifications
    """
    # Create validation result object
    result = ValidationResult()
    
    # Simple implementation for now
    logger.info("Applying validation rules to extracted data")
    
    return result

def enhance_extracted_data(data: Dict[str, Any], raw_text: str = None) -> Dict[str, Any]:
    """
    Enhance extracted data with additional processing.
    
    Args:
        data: Dictionary of extracted fields
        raw_text: Optional raw text for additional enhancement
        
    Returns:
        Enhanced data dictionary
    """
    # Create a copy to avoid modifying the original
    enhanced = data.copy()
    
    # Simple implementation for now
    logger.info("Enhancing extracted data")
    
    return enhanced
