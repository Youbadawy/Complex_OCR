"""
Medical Report Parser Module

This module provides a unified approach to extracting structured information
from medical report OCR text, specifically focusing on mammography reports.
"""

import re
import logging
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
import pandas as pd

# Setup logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ExtractedField:
    """Represents an extracted field with value and confidence"""
    value: Any
    confidence: float = 0.0
    source: str = "rule_based"  # Could be "rule_based", "llm", "template"
    
    def __post_init__(self):
        # Standardize None or empty strings to "N/A"
        if self.value is None or (isinstance(self.value, str) and not self.value.strip()):
            self.value = "N/A"
            self.confidence = 0.0

class MedicalReportParser:
    """
    Unified parser for medical reports that extracts structured data from OCR text.
    
    This class consolidates multiple extraction approaches into a single pipeline,
    focusing on accurate extraction of key fields from mammography reports.
    """
    
    # Fields that we aim to extract from the reports
    EXPECTED_FIELDS = [
        "patient_name", 
        "age", 
        "exam_date", 
        "clinical_history", 
        "findings", 
        "impression", 
        "recommendation", 
        "birads_score",
        "facility", 
        "exam_type",
        "referring_provider", 
        "interpreting_provider"
    ]
    
    def __init__(self, text: str):
        """
        Initialize the parser with OCR text.
        
        Args:
            text: Raw OCR text extracted from a medical report
        """
        self.raw_text = text
        self.preprocessed_text = self._preprocess_text(text)
        self.extracted_fields = {field: ExtractedField("N/A") for field in self.EXPECTED_FIELDS}
        
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess the OCR text to improve extraction accuracy.
        
        Args:
            text: Raw OCR text
            
        Returns:
            Preprocessed text
        """
        if not text or not isinstance(text, str):
            return ""
            
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common OCR errors
        text = text.replace('|', 'I')
        text = text.replace('l', 'I')
        text = text.replace('0', 'O')
        
        return text.strip()
        
    def extract_all_fields(self) -> Dict[str, ExtractedField]:
        """
        Extract all expected fields from the text.
        
        Returns:
            Dictionary of field names to ExtractedField objects
        """
        # Extract each field using its specific extractor
        for field in self.EXPECTED_FIELDS:
            extractor_method = getattr(self, f"_extract_{field}", None)
            if extractor_method:
                extracted = extractor_method()
                if extracted and extracted.value != "N/A":
                    self.extracted_fields[field] = extracted
        
        # Verify and enhance extractions
        self._verify_extraction()
        
        return self.extracted_fields
    
    def _extract_patient_name(self) -> ExtractedField:
        """Extract patient name from the text"""
        # Common patterns for patient name
        patterns = [
            r"Patient(?:\s*Name)?[:;]\s*([A-Za-z\s\-'.]+)",
            r"Name[:;]\s*([A-Za-z\s\-'.]+)",
            r"(?:PATIENT|PATIENT ID)[:;]\s*([A-Za-z\s\-'.]+)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, self.preprocessed_text, re.IGNORECASE)
            if match:
                value = match.group(1).strip()
                # Clean up the name
                if value and len(value) > 2:  # Avoid very short matches
                    return ExtractedField(value, 0.9)
        
        # Check if name is redacted
        redaction_patterns = [
            r"Patient(?:\s*Name)?[:;]\s*(?:REDACTED|XXX)",
            r"Name[:;]\s*(?:REDACTED|XXX)"
        ]
        
        for pattern in redaction_patterns:
            if re.search(pattern, self.preprocessed_text, re.IGNORECASE):
                return ExtractedField("Redacted", 0.8)
                
        return ExtractedField("N/A")
    
    def _extract_age(self) -> ExtractedField:
        """Extract patient age from the text"""
        # Common patterns for age
        patterns = [
            r"(?:Age|AGE)[:;]\s*(\d{1,3})",
            r"(?:Age|AGE)(?:\s*in\s*years)?[:;]?\s*(\d{1,3})",
            r"(\d{1,3})(?:\s*|-)(?:year|yr)(?:s)?(?:\s*|-)?old",
            r"(\d{1,3})(?:\s*|-)(?:yo|y/o|y.o.)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, self.preprocessed_text, re.IGNORECASE)
            if match:
                try:
                    age = int(match.group(1))
                    # Validate age is reasonable
                    if 1 <= age <= 120:
                        return ExtractedField(str(age), 0.9)
                except ValueError:
                    continue
        
        return ExtractedField("N/A")
    
    def _extract_exam_date(self) -> ExtractedField:
        """Extract examination date from the text"""
        # Common date patterns
        patterns = [
            # ISO format: YYYY-MM-DD
            (r"(?:Exam|Examination)?\s*Date[:;]?\s*(\d{4}[-/]\d{1,2}[-/]\d{1,2})", 0.9),
            
            # US format: MM/DD/YYYY
            (r"(?:Exam|Examination)?\s*Date[:;]?\s*(\d{1,2}[-/]\d{1,2}[-/]\d{4})", 0.8),
            
            # Text format: Month DD, YYYY
            (r"(?:Exam|Examination)?\s*Date[:;]?\s*([A-Za-z]+\s+\d{1,2},?\s+\d{4})", 0.8),
            
            # Electronically signed date
            (r"Electronically\s+Signed[^0-9]*(\d{1,2}[-/]\d{1,2}[-/]\d{4})", 0.7),
            (r"Electronically\s+Signed[^0-9]*(\d{4}[-/]\d{1,2}[-/]\d{1,2})", 0.7)
        ]
        
        for pattern, confidence in patterns:
            match = re.search(pattern, self.preprocessed_text, re.IGNORECASE)
            if match:
                date_str = match.group(1).strip()
                
                # Try to normalize date format if possible (simple approach)
                try:
                    # Check if it's in ISO format YYYY-MM-DD
                    if re.match(r'^\d{4}[-/]\d{1,2}[-/]\d{1,2}$', date_str):
                        # Just ensure it uses hyphens
                        date_str = date_str.replace('/', '-')
                    # Check if it's in US format MM/DD/YYYY
                    elif re.match(r'^\d{1,2}[-/]\d{1,2}[-/]\d{4}$', date_str):
                        parts = re.split(r'[-/]', date_str)
                        # Convert to YYYY-MM-DD
                        date_str = f"{parts[2]}-{parts[0].zfill(2)}-{parts[1].zfill(2)}"
                except Exception:
                    # If normalization fails, keep the original format
                    pass
                
                return ExtractedField(date_str, confidence)
        
        return ExtractedField("N/A")
    
    def _extract_clinical_history(self) -> ExtractedField:
        """Extract clinical history from the text"""
        # Common section headers for clinical history
        section_headers = [
            r"(?:Clinical\s+History|History|HISTORY|CLINICAL HISTORY)[:;]",
            r"(?:Clinical\s+Information|CLINICAL INFORMATION)[:;]",
            r"(?:Patient\s+History|PATIENT HISTORY)[:;]",
            r"(?:Medical\s+History|MEDICAL HISTORY)[:;]"
        ]
        
        # Try to find sections based on headers
        for header in section_headers:
            match = re.search(f"{header}\s*(.*?)(?:(?:{self._get_section_boundary_regex()})|$)", 
                              self.preprocessed_text, re.IGNORECASE | re.DOTALL)
            if match:
                history = match.group(1).strip()
                if history:
                    return ExtractedField(history, 0.9)
        
        # Look for history mentions in a sentence
        history_patterns = [
            r"(?:clinical\s+history|history)\s+(?:of|is|was|:)\s+([^.]+\.)",
            r"(?:Presented|Presents|Presenting)\s+with\s+([^.]+\.)"
        ]
        
        for pattern in history_patterns:
            match = re.search(pattern, self.preprocessed_text, re.IGNORECASE)
            if match:
                history = match.group(1).strip()
                if history:
                    return ExtractedField(history, 0.7)
        
        return ExtractedField("N/A")
    
    def _extract_findings(self) -> ExtractedField:
        """Extract findings (mammograph results) from the text"""
        # Common section headers for findings
        section_headers = [
            r"(?:FINDINGS|Findings)[:;]",
            r"(?:FINDING|Finding)[:;]",
            r"(?:RESULTS|Results)[:;]",
            r"(?:MAMMOGRAPHIC\s+FINDINGS|Mammographic\s+Findings)[:;]",
            r"(?:MAMMOGRAM\s+FINDINGS|Mammogram\s+Findings)[:;]"
        ]
        
        # Try to find sections based on headers
        for header in section_headers:
            match = re.search(f"{header}\s*(.*?)(?:(?:{self._get_section_boundary_regex()})|$)", 
                              self.preprocessed_text, re.IGNORECASE | re.DOTALL)
            if match:
                findings = match.group(1).strip()
                if findings:
                    return ExtractedField(findings, 0.9)
        
        return ExtractedField("N/A")
    
    def _extract_impression(self) -> ExtractedField:
        """Extract impression from the text"""
        # Common section headers for impression
        section_headers = [
            r"(?:IMPRESSION|Impression)[:;]",
            r"(?:IMPRESSIONS|Impressions)[:;]",
            r"(?:INTERPRETATION|Interpretation)[:;]",
            r"(?:ASSESSMENT|Assessment)[:;]",
            r"(?:CONCLUSION|Conclusion)[:;]"
        ]
        
        # Try to find sections based on headers
        for header in section_headers:
            match = re.search(f"{header}\s*(.*?)(?:(?:{self._get_section_boundary_regex()})|$)", 
                              self.preprocessed_text, re.IGNORECASE | re.DOTALL)
            if match:
                impression = match.group(1).strip()
                if impression:
                    return ExtractedField(impression, 0.9)
        
        return ExtractedField("N/A")
    
    def _extract_recommendation(self) -> ExtractedField:
        """Extract recommendation from the text"""
        # Common section headers for recommendation
        section_headers = [
            r"(?:RECOMMENDATION|Recommendation)[:;]",
            r"(?:RECOMMENDATIONS|Recommendations)[:;]",
            r"(?:FOLLOW-UP|Follow-up)[:;]",
            r"(?:FOLLOW UP|Follow Up)[:;]",
            r"(?:PLAN|Plan)[:;]"
        ]
        
        # Try to find sections based on headers
        for header in section_headers:
            match = re.search(f"{header}\s*(.*?)(?:(?:{self._get_section_boundary_regex()})|$)", 
                              self.preprocessed_text, re.IGNORECASE | re.DOTALL)
            if match:
                recommendation = match.group(1).strip()
                if recommendation:
                    return ExtractedField(recommendation, 0.9)
        
        # Look for recommendation in sentences
        recommendation_patterns = [
            r"(?:recommend|recommended|recommending)[^.]*?(?:in|for|after|follow-up|screening)[^.]*\.",
            r"(?:follow-up|follow up)[^.]*?(?:in|after|recommend)[^.]*\."
        ]
        
        for pattern in recommendation_patterns:
            match = re.search(pattern, self.preprocessed_text, re.IGNORECASE)
            if match:
                return ExtractedField(match.group(0).strip(), 0.7)
        
        return ExtractedField("N/A")
    
    def _extract_birads_score(self) -> ExtractedField:
        """Extract BI-RADS score from the text"""
        # Common patterns for BI-RADS
        patterns = [
            r"(?:BI-?RADS|BIRADS)\s*(?:Category|Cat\.?|assessment|score)?[:\s]+(\d+)",
            r"(?:BI-?RADS|BIRADS)[:\s]*(\d+)",
            r"(?:BI-?RADS|BIRADS)\s*(?:Category|Cat\.?)?[:\s]*(0|1|2|3|4|5|6)",
            r"Category\s+(?:BI-?RADS|BIRADS)[:\s]*(0|1|2|3|4|5|6)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, self.preprocessed_text, re.IGNORECASE)
            if match:
                birads = match.group(1).strip()
                return ExtractedField(birads, 0.9)
        
        # Look for direct statements about BI-RADS
        statement_patterns = [
            r"This is a BI-?RADS\s+(?:Category\s+)?(\d+)",
            r"assessment is BI-?RADS\s+(?:Category\s+)?(\d+)"
        ]
        
        for pattern in statement_patterns:
            match = re.search(pattern, self.preprocessed_text, re.IGNORECASE)
            if match:
                birads = match.group(1).strip()
                return ExtractedField(birads, 0.9)
                
        return ExtractedField("N/A")
    
    def _extract_facility(self) -> ExtractedField:
        """Extract facility information from the text"""
        # Common patterns for facility
        patterns = [
            r"(?:Facility|FACILITY|Location|LOCATION)[:;]\s*([A-Za-z0-9\s\-,.]+)",
            r"(?:Hospital|HOSPITAL|Clinic|CLINIC)[:;]\s*([A-Za-z0-9\s\-,.]+)",
            r"(?:Performed at|PERFORMED AT)[:;]\s*([A-Za-z0-9\s\-,.]+)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, self.preprocessed_text, re.IGNORECASE)
            if match:
                facility = match.group(1).strip()
                if facility and len(facility) > 2:  # Avoid very short matches
                    return ExtractedField(facility, 0.9)
        
        # Try to find the facility from the header (often in the first few lines)
        header_lines = '\n'.join(self.preprocessed_text.split('\n')[:10])
        header_match = re.search(r"^([A-Z][A-Za-z0-9\s\-,.&]+(?:HOSPITAL|MEDICAL CENTER|CLINIC|IMAGING))", 
                                header_lines, re.MULTILINE)
        if header_match:
            facility = header_match.group(1).strip()
            if facility and len(facility) > 5:  # Longer match for higher confidence
                return ExtractedField(facility, 0.7)
                
        return ExtractedField("N/A")
    
    def _extract_exam_type(self) -> ExtractedField:
        """Extract exam type from the text"""
        # Common patterns for mammography exam types
        mammo_patterns = [
            r"(BILATERAL\s+SCREENING\s+MAMMOGRAPHY)",
            r"(BILATERAL\s+DIAGNOSTIC\s+MAMMOGRAPHY)",
            r"(UNILATERAL\s+(?:RIGHT|LEFT)\s+DIAGNOSTIC\s+MAMMOGRAPHY)",
            r"(SCREENING\s+MAMMOGRAM)",
            r"(DIAGNOSTIC\s+MAMMOGRAM)",
            r"(MAMMOGRAM\s+(?:WITH|W/)?\s+TOMOSYNTHESIS)"
        ]
        
        for pattern in mammo_patterns:
            match = re.search(pattern, self.preprocessed_text, re.IGNORECASE)
            if match:
                exam_type = match.group(1).strip()
                return ExtractedField(exam_type, 0.9)
        
        # Check if it mentions mammogram/mammography
        if re.search(r"mammo(?:gram|graphy)", self.preprocessed_text, re.IGNORECASE):
            # Determine if screening or diagnostic
            if re.search(r"screen(?:ing)?", self.preprocessed_text, re.IGNORECASE):
                return ExtractedField("SCREENING MAMMOGRAM", 0.8)
            elif re.search(r"diagnos(?:tic|is)", self.preprocessed_text, re.IGNORECASE):
                return ExtractedField("DIAGNOSTIC MAMMOGRAM", 0.8)
            else:
                return ExtractedField("MAMMOGRAM", 0.7)
                
        return ExtractedField("N/A")
    
    def _extract_referring_provider(self) -> ExtractedField:
        """Extract referring provider from the text"""
        # Common patterns for referring provider
        patterns = [
            r"(?:Referring|REFERRING)\s+(?:Provider|PROVIDER|Physician|PHYSICIAN|Doctor|DOCTOR)[:;]?\s*([A-Za-z\s\-'.]+[A-Za-z.])",
            r"(?:Referring|REFERRING)[:;]?\s*([A-Za-z\s\-'.]+[A-Za-z.])",
            r"(?:Ordered|ORDERED)\s+(?:by|BY)[:;]?\s*([A-Za-z\s\-'.]+[A-Za-z.])",
            r"(?:Requisition|REQUISITION)[:;]?\s*(?:Dr|DR)?\.?\s*([A-Za-z\s\-'.]+[A-Za-z.])"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, self.preprocessed_text, re.IGNORECASE)
            if match:
                provider = match.group(1).strip()
                if provider and len(provider) > 3:  # Avoid very short matches
                    # Clean up standard prefixes/suffixes
                    provider = re.sub(r'^(?:Dr\.?|MD|Doctor)\s+', '', provider, flags=re.IGNORECASE)
                    provider = re.sub(r'\s+(?:MD|Ph\.?D|DO)$', '', provider, flags=re.IGNORECASE)
                    return ExtractedField(provider, 0.9)
                
        return ExtractedField("N/A")
    
    def _extract_interpreting_provider(self) -> ExtractedField:
        """Extract interpreting provider (radiologist) from the text"""
        # Common patterns for interpreting provider
        patterns = [
            r"(?:Interpreted|INTERPRETED|Read|READ)\s+(?:by|BY)[:;]?\s*([A-Za-z\s\-'.]+[A-Za-z.])",
            r"(?:Radiologist|RADIOLOGIST|Interpreter|INTERPRETER)[:;]?\s*([A-Za-z\s\-'.]+[A-Za-z.])",
            r"(?:Electronically\s+Signed|ELECTRONICALLY SIGNED)[:;]?\s*(?:by|BY)?[:;]?\s*([A-Za-z\s\-'.]+[A-Za-z.])",
            r"(?:Signed|SIGNED)[:;]?\s*(?:by|BY)?[:;]?\s*([A-Za-z\s\-'.]+[A-Za-z.])"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, self.preprocessed_text, re.IGNORECASE)
            if match:
                provider = match.group(1).strip()
                if provider and len(provider) > 3:  # Avoid very short matches
                    # Clean up standard prefixes/suffixes
                    provider = re.sub(r'^(?:Dr\.?|MD|Doctor)\s+', '', provider, flags=re.IGNORECASE)
                    provider = re.sub(r'\s+(?:MD|Ph\.?D|DO)$', '', provider, flags=re.IGNORECASE)
                    return ExtractedField(provider, 0.9)
                
        return ExtractedField("N/A")
    
    def _get_section_boundary_regex(self) -> str:
        """Get a regex pattern for common section boundaries in medical reports"""
        # Common section headers that would denote the end of a previous section
        section_headers = [
            "FINDINGS",
            "FINDING",
            "IMPRESSION",
            "IMPRESSIONS",
            "INTERPRETATION",
            "HISTORY",
            "CLINICAL HISTORY",
            "TECHNIQUE",
            "COMPARISON",
            "RECOMMENDATION",
            "RECOMMENDATIONS",
            "FOLLOW-UP",
            "FOLLOW UP",
            "ASSESSMENT",
            "CONCLUSION"
        ]
        
        # Create alternation of section headers
        section_pattern = '|'.join([f"{header}" for header in section_headers])
        return f"(?:{section_pattern})[:;]"
    
    def _verify_extraction(self) -> None:
        """Verify and enhance extracted fields"""
        # Get values safely, ensuring we handle potential dictionary values
        def get_safe_value(field_name):
            field = self.extracted_fields.get(field_name)
            if hasattr(field, 'value'):
                return field.value
            elif isinstance(field, dict) and 'value' in field:
                return field['value']
            return field if isinstance(field, str) else "N/A"
        
        # Ensure consistency between findings and impression
        findings = get_safe_value("findings")
        impression = get_safe_value("impression")
        
        # If findings is N/A but impression is available
        if findings == "N/A" and impression != "N/A":
            self.extracted_fields["findings"] = ExtractedField(impression, 0.7, "derived")
        
        # If impression is N/A but findings is available
        if impression == "N/A" and findings != "N/A":
            self.extracted_fields["impression"] = ExtractedField(findings, 0.7, "derived")
            
        # If clinical_history is N/A, check for history in the text
        if get_safe_value("clinical_history") == "N/A":
            if "history" in self.preprocessed_text.lower():
                # Try a broader extraction
                history_match = re.search(r"(?:history|presenting).{1,100}", 
                                           self.preprocessed_text, re.IGNORECASE)
                if history_match:
                    self.extracted_fields["clinical_history"] = ExtractedField(
                        history_match.group(0).strip(), 0.5, "derived")
            
    def to_dataframe_row(self) -> pd.Series:
        """Convert extracted fields to a pandas Series for DataFrame row"""
        # Create a dictionary with field values
        data = {field: self.extracted_fields[field].value for field in self.EXPECTED_FIELDS}
        
        # Add the raw text
        data["raw_ocr_text"] = self.raw_text
        
        return pd.Series(data)

    @classmethod
    def parse_batch(cls, texts: List[str]) -> pd.DataFrame:
        """
        Parse a batch of OCR texts and return as a DataFrame.
        
        Args:
            texts: List of OCR texts to parse
            
        Returns:
            DataFrame with extracted fields
        """
        rows = []
        for text in texts:
            parser = cls(text)
            parser.extract_all_fields()
            rows.append(parser.to_dataframe_row())
        
        return pd.DataFrame(rows)


# Utility function for easy access
def parse_medical_report(text: str) -> Dict[str, Any]:
    """
    Parse medical report OCR text and return structured data.
    
    Args:
        text: Raw OCR text from a medical report
        
    Returns:
        Dictionary with extracted fields
    """
    parser = MedicalReportParser(text)
    extracted = parser.extract_all_fields()
    return {field: item.value for field, item in extracted.items()} 