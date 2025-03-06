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
import time

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
        "interpreting_provider",
        "mammograph_results"
    ]
    
    def __init__(self, text: str):
        """
        Initialize the parser with OCR text.
        
        Args:
            text: Raw OCR text extracted from a medical report
        """
        self.raw_text = text
        self.preprocessed_text = self._preprocess_text(text)
        self.language = self._detect_language(self.preprocessed_text)
        self.normalized_text = self.preprocessed_text.lower()  # For case-insensitive searches
        
        # Diagnostic information
        self.metadata = {
            'language': self.language,
            'document_length': len(text) if text else 0,
            'extraction_stats': {},
            'validation_issues': []
        }
        
        # Initialize fields with N/A
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
        
        # Fix common OCR errors in specific contexts
        # Replace lowercase 'l' with uppercase 'I' in specific contexts
        text = re.sub(r'BlRADS', 'BIRADS', text)
        text = re.sub(r'Bl-RADS', 'BI-RADS', text)
        
        # Fix common letter confusions
        text = text.replace('|', 'I')  # Vertical bar to capital I
        
        # Only replace 'l' with 'I' in specific medical contexts to avoid over-correction
        medical_terms = ['BlRADS', 'cllnical', 'medlcal', 'radiologl', 'mammo']
        for term in medical_terms:
            corrected = term.replace('l', 'I')
            text = text.replace(term, corrected)
        
        # Fix zero/O confusion in specific contexts
        text = re.sub(r'(\d)O(\d)', lambda m: m.group(1) + '0' + m.group(2), text)
        
        # Fix for start of number pattern
        text = re.sub(r'(\b)O(\d)', lambda m: m.group(1) + '0' + m.group(2), text)
        
        # Remove redundant document markers that can confuse section boundaries
        text = re.sub(r'PROTÉGÉ B|PROTEGE B|PROTECTED B', '', text)
        text = re.sub(r'(?:Page|Printed on|Printed by)[^.]*?(?:\n|$)', '', text)
        
        # Create a normalized version for the instance to use in searches
        self.normalized_text = text.lower()
        
        return text.strip()
        
    def extract_all_fields(self) -> Dict[str, ExtractedField]:
        """
        Extract all expected fields from the text.
        
        Returns:
            Dictionary of field names to ExtractedField objects
        """
        # Record start time for performance monitoring
        start_time = time.time()
        
        # Extract each field using its specific extractor
        for field in self.EXPECTED_FIELDS:
            extractor_method = getattr(self, f"_extract_{field}", None)
            if extractor_method:
                try:
                    extracted = extractor_method()
                    if extracted:
                        self.extracted_fields[field] = extracted
                except Exception as e:
                    logger.error(f"Error extracting {field}: {str(e)}")
                    self.metadata['validation_issues'].append(f"Error extracting {field}: {str(e)}")
        
        # Verify and enhance extractions
        self._verify_extraction()
        
        # Final cleanup - standardize formatting
        for field, item in self.extracted_fields.items():
            if isinstance(item.value, str):
                # If there are leading/trailing quotes, remove them
                value = item.value.strip('"\'')
                
                # Fix spacing
                value = re.sub(r'\s+', ' ', value).strip()
                
                # Update the value
                self.extracted_fields[field] = ExtractedField(
                    value, item.confidence, item.source
                )
        
        # Record extraction time
        self.metadata['extraction_time'] = time.time() - start_time
        self.metadata['fields_extracted'] = sum(1 for f in self.extracted_fields.values() if f.value != "N/A")
        
        return self.extracted_fields
    
    def _extract_patient_name(self) -> ExtractedField:
        """Extract patient name from the text"""
        # Check first if document indicates redacted information
        redaction_patterns = [
            r"Patient(?:\s*Name)?[:;]\s*(?:REDACTED|XXX|[*]+)",
            r"Name[:;]\s*(?:REDACTED|XXX|[*]+)",
            r"Patient[\s:]+(?:information restricted|restricted|redacted)"
        ]
        
        for pattern in redaction_patterns:
            if re.search(pattern, self.preprocessed_text, re.IGNORECASE):
                return ExtractedField("Redacted", 0.9, "pattern_match")
        
        # Define patterns based on language
        if self.language == 'fr':
            patterns = [
                r"(?:Patient|Patiente|Nom du patient)[:;]\s*([A-Za-z\s\-'.]{2,30}?)(?:\s*\n|\s{2,}|$)",
                r"(?:Nom|Nom:)[:;]\s*([A-Za-z\s\-'.]{2,30}?)(?:\s*\n|\s{2,}|$)",
                r"(?:PATIENT|NOM)[:;]\s*([A-Za-z\s\-'.]{2,30}?)(?:\s*\n|\s{2,}|$)"
            ]
        else:  # English patterns
            patterns = [
                r"Patient(?:\s*Name)?[:;]\s*([A-Za-z\s\-'.]{2,30}?)(?:\s*\n|\s{2,}|$)",
                r"Name[:;]\s*([A-Za-z\s\-'.]{2,30}?)(?:\s*\n|\s{2,}|$)",
                r"(?:PATIENT|PATIENT ID)[:;]\s*([A-Za-z\s\-'.]{2,30}?)(?:\s*\n|\s{2,}|$)"
            ]
        
        # Try each pattern
        for pattern in patterns:
            match = re.search(pattern, self.preprocessed_text, re.IGNORECASE)
            if match:
                name_candidate = match.group(1).strip()
                
                # Filter out false positives
                if self._validate_patient_name(name_candidate):
                    self.metadata['extraction_stats']['patient_name'] = f"Matched pattern: {pattern}"
                    return ExtractedField(name_candidate, 0.9, "pattern_match")
                
        # Try to find patient name in specific document sections
        # Look for name near the beginning of the document, after "Patient:" header
        doc_start = self.preprocessed_text[:min(500, len(self.preprocessed_text))]
        name_section_match = re.search(r'Patient:?\s*(.{2,40}?)(?:\n|Gender|\s{3,}|$)', doc_start, re.IGNORECASE)
        if name_section_match:
            name_candidate = name_section_match.group(1).strip()
            if self._validate_patient_name(name_candidate):
                self.metadata['extraction_stats']['patient_name'] = "Found in document start section"
                return ExtractedField(name_candidate, 0.8, "section_match")
        
        # If we got this far, we couldn't find a valid patient name
        self.metadata['validation_issues'].append("No valid patient name found")
        return ExtractedField("N/A", 0.0, "not_found")
    
    def _validate_patient_name(self, name_candidate: str) -> bool:
        """
        Validate if a string is likely to be a patient name.
        
        Args:
            name_candidate: String to validate as patient name
            
        Returns:
            Boolean indicating if this is likely a valid patient name
        """
        # Check length
        if len(name_candidate) < 2 or len(name_candidate) > 40:
            return False
        
        # Filter out document headers that often get mistaken for names
        header_indicators = [
            'document date', 'scan', 'mammo', 'breast', 'ultrasound', 
            'bilat', 'report', 'sein', 'patient:', 'name:', 'nom:',
            'exam', 'examen', 'printed', 'page'
        ]
        
        # Check if the candidate contains header indicators
        lower_candidate = name_candidate.lower()
        if any(indicator in lower_candidate for indicator in header_indicators):
            return False
        
        # Check for all-caps (often headers, not names)
        if name_candidate.isupper() and len(name_candidate) > 10:
            return False
        
        # Names typically have at least one letter
        if not re.search(r'[A-Za-z]', name_candidate):
            return False
        
        # Names don't typically have numbers
        if re.search(r'\d', name_candidate):
            return False
        
        # Must pass all validation
        return True
    
    def _extract_age(self) -> ExtractedField:
        """Extract patient age from the text"""
        # Common explicit age patterns
        age_patterns = [
            # Explicit age indicators with numbers
            r"(?:Age|AGE)[:;]\s*(\d{1,3})",
            r"(?:Age|AGE)(?:\s*in\s*years)?[:;]?\s*(\d{1,3})",
            r"(\d{1,3})(?:\s*|-)(?:year|yr)(?:s)?(?:\s*|-)?old",
            r"(\d{1,3})(?:\s*|-)(?:yo|y/o|y.o.)",
            
            # French patterns
            r"(?:Âge|Age)[:;]\s*(\d{1,3})",
            r"(?:Âge|Age)(?:\s*en\s*années)?[:;]?\s*(\d{1,3})",
            r"(\d{1,3})(?:\s*|-)(?:ans|an)(?:\s*|-)?",
        ]
        
        # Look for age in medical history - common pattern, e.g., "mother with breast cancer at age 55"
        context_patterns = [
            r"(?:at|at the|at age|à l'âge de)\s+(?:age\s+)?(\d{1,3})",
            r"(?:aged|âgée de)\s+(\d{1,3})"
        ]
        
        # Context where these ages could be confused with page numbers
        negation_contexts = [
            r"page\s+\d+\s+of\s+\d+", 
            r"printed", 
            r"page:", 
            r"p.\s+\d+",
            r"\d{4}-\d{1,2}-\d{1,2}",  # Date patterns
            r"\d{1,2}/\d{1,2}/\d{4}"   # Date patterns
        ]
        
        # Try pattern-based extraction first (more reliable)
        for pattern in age_patterns:
            match = re.search(pattern, self.preprocessed_text, re.IGNORECASE)
            if match:
                try:
                    age = int(match.group(1))
                    
                    # Validate age is reasonable for a mammography patient (18-120)
                    if 18 <= age <= 120:
                        # Check if this match is not within a negation context
                        match_pos = match.start()
                        surrounding_text = self.preprocessed_text[max(0, match_pos-20):min(len(self.preprocessed_text), match_pos+20)]
                        
                        if not any(re.search(neg, surrounding_text, re.IGNORECASE) for neg in negation_contexts):
                            self.metadata['extraction_stats']['age'] = f"Matched age pattern: {pattern}"
                            return ExtractedField(str(age), 0.9, "pattern_match")
                except ValueError:
                    continue
        
        # Check for age in family history context
        history_text = self._get_section("clinical_history")
        if history_text:
            for pattern in context_patterns:
                matches = re.finditer(pattern, history_text, re.IGNORECASE)
                for match in matches:
                    try:
                        age = int(match.group(1))
                        # Filter out unreasonable ages
                        if 18 <= age <= 120:
                            # Don't extract family member ages as patient age
                            context_start = max(0, match.start() - 30)
                            context = history_text[context_start:match.start()]
                            if not re.search(r'(?:mother|father|sister|brother|aunt|uncle|cousin|family)', context, re.IGNORECASE):
                                self.metadata['extraction_stats']['age'] = f"Found in history context"
                                return ExtractedField(str(age), 0.7, "context_match")
                    except ValueError:
                        continue
        
        # No valid age found
        self.metadata['validation_issues'].append("No valid age found")
        return ExtractedField("N/A", 0.0, "not_found")
    
    def _get_section(self, section_name: str) -> str:
        """
        Get a section of text from the document based on common section headers.
        
        Args:
            section_name: Name of the section to extract
            
        Returns:
            Extracted section text or empty string if not found
        """
        section_patterns = {
            "clinical_history": [
                r"(?:Clinical\s+History|History|HISTORY|CLINICAL HISTORY)[:;]\s*(.*?)(?=(?:FINDINGS|IMPRESSION|TECHNIQUE|COMPARISON|RECOMMENDATION|ASSESSMENT|CONCLUSION)|$)",
                r"(?:Clinical\s+Information|CLINICAL INFORMATION)[:;]\s*(.*?)(?=(?:FINDINGS|IMPRESSION|TECHNIQUE|COMPARISON|RECOMMENDATION|ASSESSMENT|CONCLUSION)|$)",
                r"(?:Patient\s+History|PATIENT HISTORY)[:;]\s*(.*?)(?=(?:FINDINGS|IMPRESSION|TECHNIQUE|COMPARISON|RECOMMENDATION|ASSESSMENT|CONCLUSION)|$)",
                
                # French patterns
                r"(?:Antécédents|Histoire\s+clinique|HISTOIRE CLINIQUE)[:;]\s*(.*?)(?=(?:RÉSULTATS|IMPRESSION|TECHNIQUE|COMPARAISON|RECOMMANDATION|ÉVALUATION|CONCLUSION)|$)",
            ],
            "findings": [
                r"(?:FINDINGS|Findings)[:;]\s*(.*?)(?=(?:IMPRESSION|ASSESSMENT|RECOMMENDATION|CONCLUSION)|$)",
                r"(?:MAMMOGRAM|MAMMOGRAPHY|ULTRASOUND)[:;]\s*(.*?)(?=(?:IMPRESSION|ASSESSMENT|RECOMMENDATION|CONCLUSION)|$)",
                
                # French patterns
                r"(?:RÉSULTATS|Résultats|CONSTATATIONS|Constatations)[:;]\s*(.*?)(?=(?:IMPRESSION|ÉVALUATION|RECOMMANDATION|CONCLUSION)|$)",
            ],
            # Add other sections as needed
        }
        
        if section_name in section_patterns:
            for pattern in section_patterns[section_name]:
                match = re.search(pattern, self.preprocessed_text, re.IGNORECASE | re.DOTALL)
                if match:
                    return match.group(1).strip()
        
        return ""
    
    def _extract_exam_date(self) -> ExtractedField:
        """Extract examination date from the document"""
        # Initialize with default values
        exam_date = "N/A"
        confidence = 0.0
        match_type = "not_found"
        
        # Different patterns based on language
        if self.language == "fr":
            # French patterns
            direct_patterns = [
                r"(?:Date d'examen|Date de l'examen)[:;]\s*(\d{1,2}[-/\s.]\d{1,2}[-/\s.]\d{2,4}|\d{1,2}\s+(?:janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)\s+\d{2,4})",
                r"(?:Date du document|Date du rapport)[:;]\s*(\d{1,2}[-/\s.]\d{1,2}[-/\s.]\d{2,4}|\d{1,2}\s+(?:janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)\s+\d{2,4})",
                r"(?:Examen réalisé le|Effectué le)[:;]?\s*(\d{1,2}[-/\s.]\d{1,2}[-/\s.]\d{2,4}|\d{1,2}\s+(?:janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)\s+\d{2,4})"
            ]
        else:
            # English patterns
            direct_patterns = [
                r"(?:Exam Date|Examination Date)[:;]\s*(\d{1,2}[-/\s.]\d{1,2}[-/\s.]\d{2,4}|\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4})",
                r"(?:Document Date|Report Date)[:;]\s*(\d{1,2}[-/\s.]\d{1,2}[-/\s.]\d{2,4}|\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4})",
                r"(?:Performed on|Study Date)[:;]?\s*(\d{1,2}[-/\s.]\d{1,2}[-/\s.]\d{2,4}|\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4})"
            ]
        
        # Try direct patterns first
        for pattern in direct_patterns:
            match = re.search(pattern, self.preprocessed_text, re.IGNORECASE)
            if match:
                exam_date = match.group(1).strip()
                confidence = 0.9
                match_type = "direct_pattern"
                break
        
        # If no direct match, try to find dates in the document header
        if exam_date == "N/A":
            # Look for dates in the first 10 lines of the document
            header_lines = self.preprocessed_text.split('\n')[:10]
            header_text = '\n'.join(header_lines)
            
            # Common date formats
            date_patterns = [
                # DD/MM/YYYY or MM/DD/YYYY
                r"(\d{1,2}[-/\s.]\d{1,2}[-/\s.]\d{4})",
                # Month DD, YYYY
                r"((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4})",
                # DD Month YYYY
                r"(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4})",
                # French dates
                r"(\d{1,2}\s+(?:janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)\s+\d{4})"
            ]
            
            for pattern in date_patterns:
                matches = re.findall(pattern, header_text, re.IGNORECASE)
                if matches:
                    # Use the first date found
                    exam_date = matches[0]
                    confidence = 0.7
                    match_type = "header_date"
                    break
        
        # If a date was found, normalize it
        if exam_date != "N/A":
            try:
                normalized_date = self._normalize_date_format(exam_date)
                if normalized_date:
                    exam_date = normalized_date
                    confidence = max(confidence, 0.8)  # Increase confidence if normalization succeeded
            except Exception as e:
                self.metadata['validation_issues'].append(f"Error normalizing date: {str(e)}")
        
        return ExtractedField(exam_date, confidence, match_type)
    
    def _extract_clinical_history(self) -> ExtractedField:
        """Extract clinical history from the document"""
        # Initialize with default values
        history = "N/A"
        confidence = 0.0
        match_type = "not_found"
        
        # Different patterns based on language
        if self.language == "fr":
            # French patterns
            direct_patterns = [
                r"(?:RENSEIGNEMENTS CLINIQUES|HISTOIRE CLINIQUE|ANTÉCÉDENTS|HISTORIQUE)[:;]\s*(.*?)(?=(?:TECHNIQUE|PROTOCOLE|PROCÉDURE|RÉSULTATS|OBSERVATIONS|$))",
                r"(?:INDICATION|INDICATIONS)[:;]\s*(.*?)(?=(?:TECHNIQUE|PROTOCOLE|PROCÉDURE|RÉSULTATS|OBSERVATIONS|$))"
            ]
            section_names = ["RENSEIGNEMENTS CLINIQUES", "HISTOIRE CLINIQUE", "ANTÉCÉDENTS", "HISTORIQUE", "INDICATION"]
            history_keywords = ["antécédent", "histoire", "clinique", "indication", "symptôme"]
        else:
            # English patterns
            direct_patterns = [
                r"(?:CLINICAL HISTORY|HISTORY|CLINICAL INFORMATION|INDICATION)[:;]\s*(.*?)(?=(?:TECHNIQUE|PROCEDURE|PROTOCOL|FINDINGS|RESULT|$))",
                r"(?:REASON FOR EXAM|REASON FOR EXAMINATION)[:;]\s*(.*?)(?=(?:TECHNIQUE|PROCEDURE|PROTOCOL|FINDINGS|RESULT|$))"
            ]
            section_names = ["CLINICAL HISTORY", "HISTORY", "CLINICAL INFORMATION", "INDICATION"]
            history_keywords = ["history", "clinical", "indication", "symptom", "reason"]
        
        # Try direct patterns first
        for pattern in direct_patterns:
            match = re.search(pattern, self.preprocessed_text, re.IGNORECASE | re.DOTALL)
            if match:
                history = match.group(1).strip()
                confidence = 0.9
                match_type = "direct_pattern"
                break
        
        # If no direct match, try to get the clinical history section
        if history == "N/A":
            for section_name in section_names:
                section_content = self._get_section(section_name)
                if section_content and section_content != "N/A":
                    history = section_content
                    confidence = 0.8
                    match_type = "section_extraction"
                    break
        
        # If still not found, look for sentences containing history keywords
        if history == "N/A":
            # Split text into sentences
            sentences = re.split(r'(?<=[.!?])\s+', self.preprocessed_text)
            history_sentences = []
            
            for sentence in sentences[:10]:  # Check only first 10 sentences
                lower_sentence = sentence.lower()
                if any(keyword in lower_sentence for keyword in history_keywords):
                    history_sentences.append(sentence)
            
            if history_sentences:
                history = " ".join(history_sentences)
                confidence = 0.6
                match_type = "keyword_sentences"
        
        # Clean up the history
        if history != "N/A":
            # Remove document metadata that might be mixed in with clinical history
            history = re.sub(r'(?i)(?:Scan|Document|Report|Page|Date|Time|Patient ID|MRN)[:;].*?(?:\n|$)', '', history)
            history = re.sub(r'(?i)(?:Electronically signed|Dictated|Transcribed).*?(?:\n|$)', '', history)
            
            # Clean up the history using the standard method
            history = self._clean_section_content(history, "clinical_history")
            
            # If history is too short after cleaning, it might be invalid
            if len(history) < 5:
                self.metadata['validation_issues'].append("Clinical history is suspiciously short after cleaning")
                history = "N/A"
                confidence = 0.0
                match_type = "invalid_after_cleaning"
        
        return ExtractedField(history, confidence, match_type)
    
    def _clean_section_content(self, content: str, section_type: str) -> str:
        """
        Clean up section content by removing irrelevant information and formatting.
        
        Args:
            content: The raw section content to clean
            section_type: The type of section (findings, impression, etc.)
            
        Returns:
            Cleaned section content
        """
        if not content or content == "N/A":
            return content
        
        # Remove common OCR artifacts and formatting issues
        cleaned = content.strip()
        
        # Remove page numbers and headers/footers
        cleaned = re.sub(r'(?i)Page \d+ of \d+', '', cleaned)
        cleaned = re.sub(r'(?i)(?:Report|Document) ID:.*?(?:\n|$)', '', cleaned)
        
        # Remove document metadata that appears in sections
        metadata_patterns = [
            r'(?i)(?:Scan|Document|Report|Page|Date|Time|Patient ID|MRN)[:;].*?(?:\n|$)',
            r'(?i)(?:Electronically signed|Dictated|Transcribed).*?(?:\n|$)',
            r'(?i)(?:Printed|Generated) on.*?(?:\n|$)',
            r'(?i)(?:Confidential|Confidentiality).*?(?:\n|$)',
            r'(?i)(?:Accession|Exam|Study) (?:Number|ID|#).*?(?:\n|$)'
        ]
        
        for pattern in metadata_patterns:
            cleaned = re.sub(pattern, '', cleaned)
        
        # Language-specific cleaning
        if self.language == 'fr':
            # Remove French-specific metadata
            cleaned = re.sub(r'(?i)(?:Numéro|Dossier|Patient|Examen)[:;].*?(?:\n|$)', '', cleaned)
            cleaned = re.sub(r'(?i)(?:Signé électroniquement|Dicté|Transcrit).*?(?:\n|$)', '', cleaned)
        
        # Section-specific cleaning
        if section_type == "findings" or section_type == "impression":
            # Remove references to other sections
            cleaned = re.sub(r'(?i)(?:See|Refer to) (?:above|below|impression|findings|history).*?(?:\n|$)', '', cleaned)
            
            # Remove measurement units that might be split across lines
            cleaned = re.sub(r'\n(?:cm|mm|Hz|dB)\s', ' \\1 ', cleaned)
        
        elif section_type == "clinical_history":
            # Remove age references that might be confused with clinical history
            cleaned = re.sub(r'(?i)\b(?:age|aged)\s+\d+\b', '', cleaned)
            
            # Remove document type references
            cleaned = re.sub(r'(?i)(?:mammogram|ultrasound|MRI|breast imaging)(?:\s+report)?\s*$', '', cleaned)
        
        elif section_type == "recommendation":
            # Remove standard disclaimer text often found in recommendations
            cleaned = re.sub(r'(?i)This report has been read and approved.*?(?:\n|$)', '', cleaned)
            cleaned = re.sub(r'(?i)The results of this examination have been discussed.*?(?:\n|$)', '', cleaned)
        
        # General cleaning for all sections
        
        # Remove excessive whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # Remove isolated punctuation
        cleaned = re.sub(r'\s[.,:;]\s', ' ', cleaned)
        
        # Remove lines that are just punctuation or very short
        cleaned = re.sub(r'^\s*[.,:;-]+\s*$', '', cleaned)
        
        # Remove lines that are just single characters (likely OCR errors)
        cleaned = re.sub(r'^\s*.\s*$', '', cleaned)
        
        # Final trim and normalize spaces
        cleaned = cleaned.strip()
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        return cleaned
    
    def _extract_findings(self) -> ExtractedField:
        """Extract findings from the document"""
        # Initialize with default values
        findings = "N/A"
        confidence = 0.0
        match_type = "not_found"
        
        try:
            # Different patterns based on language
            if self.language == "fr":
                # French patterns - fixed to avoid group reference issues
                direct_patterns = [
                    r"(?:RÉSULTATS|Résultats|OBSERVATIONS|CONSTATATIONS|TROUVAILLES)[:;]\s*(.*?)(?=(?:IMPRESSION|CONCLUSION|INTERPRÉTATION|RECOMMANDATION|SUIVI|$))",
                    r"(?:RÉSULTATS DE L'EXAMEN|RÉSULTATS D'EXAMEN)[:;]\s*(.*?)(?=(?:IMPRESSION|CONCLUSION|INTERPRÉTATION|RECOMMANDATION|SUIVI|$))"
                ]
            else:
                # English patterns
                direct_patterns = [
                    r"(?:FINDINGS|FINDING|RESULT|RESULTS|OBSERVATION)[:;]\s*(.*?)(?=(?:IMPRESSION|CONCLUSION|INTERPRETATION|RECOMMENDATION|FOLLOW-UP|ASSESSMENT|$))",
                    r"(?:EXAMINATION FINDINGS|EXAM FINDINGS)[:;]\s*(.*?)(?=(?:IMPRESSION|CONCLUSION|INTERPRETATION|RECOMMENDATION|FOLLOW-UP|ASSESSMENT|$))"
                ]
            
            # Try direct patterns first
            for pattern in direct_patterns:
                match = re.search(pattern, self.text, re.IGNORECASE | re.DOTALL)
                if match:
                    findings = match.group(1).strip()
                    confidence = 0.9
                    match_type = "direct_pattern"
                    break
            
            # If no direct match, try to get the findings section
            if findings == "N/A":
                if self.language == "fr":
                    section_names = ["RÉSULTATS", "OBSERVATIONS", "CONSTATATIONS", "TROUVAILLES"]
                else:
                    section_names = ["FINDINGS", "RESULT", "OBSERVATION"]
                    
                for section_name in section_names:
                    section_content = self._get_section(section_name)
                    if section_content and section_content != "N/A":
                        findings = section_content
                        confidence = 0.8
                        match_type = "section_extraction"
                        break
            
            # Clean up the findings
            if findings != "N/A":
                findings = self._clean_section_content(findings, "findings")
                
                # If findings is too short after cleaning, it might be invalid
                if len(findings) < 10:
                    self.metadata['validation_issues'].append("Findings section is suspiciously short after cleaning")
                    if len(findings) < 5:  # Extremely short, likely invalid
                        findings = "N/A"
                        confidence = 0.0
                        match_type = "invalid_after_cleaning"
        
        except Exception as e:
            self.metadata['validation_issues'].append(f"Error extracting findings: {str(e)}")
            logging.error(f"Error extracting findings: {str(e)}")
            findings = "N/A"
            confidence = 0.0
            match_type = "extraction_error"
        
        return ExtractedField(findings, confidence, match_type)

    def _extract_impression(self) -> ExtractedField:
        """Extract impression/conclusion from the document"""
        # Initialize with default values
        impression = "N/A"
        confidence = 0.0
        match_type = "not_found"
        
        try:
            # Different patterns based on language
            if self.language == "fr":
                # French patterns - fixed to avoid group reference issues
                direct_patterns = [
                    r"(?:IMPRESSION|CONCLUSION|INTERPRÉTATION|OPINION)[:;]\s*(.*?)(?=(?:RECOMMANDATION|SUIVI|CONDUITE À TENIR|BIRADS|CATÉGORIE|$))",
                    r"(?:CONCLUSION DE L'EXAMEN|CONCLUSION D'EXAMEN)[:;]\s*(.*?)(?=(?:RECOMMANDATION|SUIVI|CONDUITE À TENIR|BIRADS|CATÉGORIE|$))"
                ]
                section_names = ["IMPRESSION", "CONCLUSION", "INTERPRÉTATION", "OPINION"]
            else:
                # English patterns
                direct_patterns = [
                    r"(?:IMPRESSION|CONCLUSION|INTERPRETATION|ASSESSMENT)[:;]\s*(.*?)(?=(?:RECOMMENDATION|FOLLOW-UP|PLAN|BIRADS|CATEGORY|$))",
                    r"(?:EXAM IMPRESSION|EXAMINATION IMPRESSION)[:;]\s*(.*?)(?=(?:RECOMMENDATION|FOLLOW-UP|PLAN|BIRADS|CATEGORY|$))"
                ]
                section_names = ["IMPRESSION", "CONCLUSION", "INTERPRETATION", "ASSESSMENT"]
            
            # Try direct patterns first
            for pattern in direct_patterns:
                match = re.search(pattern, self.text, re.IGNORECASE | re.DOTALL)
                if match:
                    impression = match.group(1).strip()
                    confidence = 0.9
                    match_type = "direct_pattern"
                    break
            
            # If no direct match, try to get the impression section
            if impression == "N/A":
                for section_name in section_names:
                    section_content = self._get_section(section_name)
                    if section_content and section_content != "N/A":
                        impression = section_content
                        confidence = 0.8
                        match_type = "section_extraction"
                        break
            
            # Clean up the impression
            if impression != "N/A":
                impression = self._clean_section_content(impression, "impression")
                
                # If impression is too short after cleaning, it might be invalid
                if len(impression) < 10:
                    self.metadata['validation_issues'].append("Impression section is suspiciously short after cleaning")
                    if len(impression) < 5:  # Extremely short, likely invalid
                        impression = "N/A"
                        confidence = 0.0
                        match_type = "invalid_after_cleaning"
        
        except Exception as e:
            self.metadata['validation_issues'].append(f"Error extracting impression: {str(e)}")
            logging.error(f"Error extracting impression: {str(e)}")
            impression = "N/A"
            confidence = 0.0
            match_type = "extraction_error"
        
        return ExtractedField(impression, confidence, match_type)
    
    def _extract_recommendation(self) -> ExtractedField:
        """Extract recommendation from the text"""
        # Directly look for Recommendation: pattern
        direct_pattern = r"Recommendation:?\s*(.*?)(?=(?:Page|END OF DOCUMENT|PROTEGE|PROTECTED|Signed|Electronically|^\s*$)|$)"
        match = re.search(direct_pattern, self.preprocessed_text, re.IGNORECASE | re.DOTALL)
        if match:
            content = match.group(1).strip()
            if content:
                cleaned_text = self._clean_section_content(content, "recommendation")
                if cleaned_text:
                    self.metadata['extraction_stats']['recommendation'] = "Found via direct pattern"
                    return ExtractedField(cleaned_text, 0.95, "direct_pattern_match")
        
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
                    cleaned_text = self._clean_section_content(recommendation, "recommendation")
                    return ExtractedField(cleaned_text, 0.9, "section_match")
        
        # Look for recommendation in sentences with key phrases like "follow-up in X months/years"
        recommendation_patterns = [
            r"(?:recommend|recommended|recommending)[^.]*?(?:in|for|after|follow-up|screening)[^.]*\.",
            r"(?:follow-up|follow up)[^.]*?(?:in|after|recommend)[^.]*\.",
            r"(?:follow-up|follow up)[^.]*?(?:\d+\s+(?:month|year|week))[^.]*\."
        ]
        
        for pattern in recommendation_patterns:
            match = re.search(pattern, self.preprocessed_text, re.IGNORECASE)
            if match:
                recommendation = match.group(0).strip()
                cleaned_text = self._clean_section_content(recommendation, "recommendation")
                return ExtractedField(cleaned_text, 0.8, "sentence_match")
        
        # If an impression suggests follow-up, use that
        impression = self._get_section("impression")
        if impression and any(term in impression.lower() for term in ["follow-up", "recommend", "screening in", "year", "month"]):
            # Extract the sentence with the recommendation
            sentences = re.split(r'[.!?]\s+', impression)
            for sentence in sentences:
                if any(term in sentence.lower() for term in ["follow-up", "recommend", "screening in", "year", "month"]):
                    cleaned_text = self._clean_section_content(sentence, "recommendation")
                    return ExtractedField(cleaned_text + ".", 0.7, "derived_from_impression")
        
        return ExtractedField("N/A", 0.0, "not_found")
    
    def _extract_birads_score(self) -> ExtractedField:
        """Extract BI-RADS score from the text"""
        # Look for explicit "BIRADS X" pattern
        direct_birads_pattern = r"(?:BIRADS|BI-?RADS)[:\s]*(\d)"
        match = re.search(direct_birads_pattern, self.preprocessed_text, re.IGNORECASE)
        if match:
            birads = match.group(1).strip()
            return ExtractedField(birads, 0.95, "direct_birads_pattern")
        
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
                return ExtractedField(birads, 0.9, "standard_pattern_match")
        
        # Look for direct statements about BI-RADS
        statement_patterns = [
            r"This is a BI-?RADS\s+(?:Category\s+)?(\d+)",
            r"assessment is BI-?RADS\s+(?:Category\s+)?(\d+)"
        ]
        
        for pattern in statement_patterns:
            match = re.search(pattern, self.preprocessed_text, re.IGNORECASE)
            if match:
                birads = match.group(1).strip()
                return ExtractedField(birads, 0.9, "statement_pattern_match")
        
        # Look for BIRADS in recommendation section
        recommendation = self._get_section("recommendation")
        if recommendation:
            birads_in_rec = re.search(r"BIRADS\s+(\d)", recommendation, re.IGNORECASE)
            if birads_in_rec:
                birads = birads_in_rec.group(1).strip()
                return ExtractedField(birads, 0.85, "recommendation_section_match")
            
        return ExtractedField("N/A", 0.0, "not_found")
    
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
        # Look for exam type in uppercase (common format)
        uppercase_exam_patterns = [
            r"(BILATERAL\s+(?:SCREENING|DIAGNOSTIC)\s+MAMMOGRAPHY(?:\s+WITH\s+TOMOSYNTHESIS)?(?:,\s*ULTRASOUND\s+(?:LEFT|RIGHT|BILATERAL)\s+BREAST)?)",
            r"((?:SCREENING|DIAGNOSTIC)\s+MAMMOGRAM(?:\s+WITH\s+TOMOSYNTHESIS)?(?:,\s*ULTRASOUND\s+(?:LEFT|RIGHT|BILATERAL)\s+BREAST)?)",
            r"(BILATERAL\s+MAMMOGRAM(?:\s+WITH\s+TOMOSYNTHESIS)?(?:,\s*ULTRASOUND\s+(?:LEFT|RIGHT|BILATERAL)\s+BREAST)?)",
            r"(MAMMOGRAM\s+(?:WITH|W/)?\s+TOMOSYNTHESIS)"
        ]
        
        # Check each pattern
        for pattern in uppercase_exam_patterns:
            match = re.search(pattern, self.preprocessed_text, re.IGNORECASE)
            if match:
                exam_type = match.group(1).strip()
                self.metadata['extraction_stats']['exam_type'] = "Found via uppercase pattern"
                return ExtractedField(exam_type, 0.95, "uppercase_pattern")
        
        # Check direct headers that often precede the exam type
        header_patterns = [
            r"Exam(?:\s+Type)?:?\s*(.*?)(?:\n|$)",
            r"Examination:?\s*(.*?)(?:\n|$)",
            r"Procedure:?\s*(.*?)(?:\n|$)"
        ]
        
        for pattern in header_patterns:
            match = re.search(pattern, self.preprocessed_text, re.IGNORECASE)
            if match:
                exam_type = match.group(1).strip()
                # Validate it looks like a mammogram description
                if any(term in exam_type.lower() for term in ['mammo', 'breast', 'tomo', 'screening', 'diagnostic']):
                    self.metadata['extraction_stats']['exam_type'] = "Found via header pattern"
                    return ExtractedField(exam_type, 0.9, "header_pattern")
        
        # Look for mammogram types in the first 1000 characters (often in document header)
        doc_start = self.preprocessed_text[:min(1000, len(self.preprocessed_text))]
        
        # Common mammography-related terms
        mammo_types = [
            r"(BILATERAL\s+SCREENING\s+MAMMOGRAPHY)",
            r"(BILATERAL\s+DIAGNOSTIC\s+MAMMOGRAPHY)",
            r"(UNILATERAL\s+(?:RIGHT|LEFT)\s+DIAGNOSTIC\s+MAMMOGRAPHY)",
            r"(SCREENING\s+MAMMOGRAM)",
            r"(DIAGNOSTIC\s+MAMMOGRAM)",
            r"(MAMMOGRAM\s+(?:WITH|W/)?\s+TOMOSYNTHESIS)",
            r"(ULTRASOUND\s+(?:LEFT|RIGHT|BILATERAL)\s+BREAST)"
        ]
        
        for pattern in mammo_types:
            match = re.search(pattern, doc_start, re.IGNORECASE)
            if match:
                exam_type = match.group(1).strip()
                self.metadata['extraction_stats']['exam_type'] = "Found in document start"
                return ExtractedField(exam_type, 0.85, "doc_start_match")
        
        # Check if it mentions mammogram/mammography anywhere
        if re.search(r"mammo(?:gram|graphy)", self.preprocessed_text, re.IGNORECASE):
            # Determine if screening or diagnostic
            if re.search(r"screen(?:ing)?", self.preprocessed_text, re.IGNORECASE):
                self.metadata['extraction_stats']['exam_type'] = "Derived from mentions"
                return ExtractedField("SCREENING MAMMOGRAM", 0.7, "derived_from_mentions")
            elif re.search(r"diagnos(?:tic|is)", self.preprocessed_text, re.IGNORECASE):
                self.metadata['extraction_stats']['exam_type'] = "Derived from mentions"
                return ExtractedField("DIAGNOSTIC MAMMOGRAM", 0.7, "derived_from_mentions")
            else:
                self.metadata['extraction_stats']['exam_type'] = "Derived from mentions"
                return ExtractedField("MAMMOGRAM", 0.6, "derived_from_mentions")
            
        return ExtractedField("N/A", 0.0, "not_found")
    
    def _extract_referring_provider(self) -> ExtractedField:
        """Extract referring provider from the text"""
        # Look for DR. NAME at the beginning of the document (often the referring doctor)
        doc_start = self.preprocessed_text[:min(500, len(self.preprocessed_text))]
        referring_dr_pattern = r"(?:DR\.|Dr\.|DOCTOR)\s+([A-Z][A-Za-z\s\-'.]+)"
        match = re.search(referring_dr_pattern, doc_start)
        if match:
            provider = match.group(1).strip()
            if provider and len(provider) > 2 and self._validate_provider_name(provider):
                self.metadata['extraction_stats']['referring_provider'] = "Found via DR. pattern at start"
                return ExtractedField(provider, 0.9, "doctor_start_pattern")
        
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
                if provider and len(provider) > 3 and self._validate_provider_name(provider):  # Avoid very short matches
                    # Clean up standard prefixes/suffixes
                    provider = re.sub(r'^(?:Dr\.?|MD|Doctor)\s+', '', provider, flags=re.IGNORECASE)
                    provider = re.sub(r'\s+(?:MD|Ph\.?D|DO)$', '', provider, flags=re.IGNORECASE)
                    self.metadata['extraction_stats']['referring_provider'] = "Found via standard pattern"
                    return ExtractedField(provider, 0.9, "standard_pattern")
                
        return ExtractedField("N/A", 0.0, "not_found")
    
    def _validate_provider_name(self, name: str) -> bool:
        """Validate if a string is likely to be a provider name
        
        Args:
            name: Potential provider name
            
        Returns:
            True if likely a valid provider name
        """
        # Check length
        if len(name) < 2 or len(name) > 40:
            return False
        
        # Check for non-name indicators
        non_name_indicators = [
            'page', 'patient', 'date', 'exam', 'report', 'printed', 
            'hospital', 'clinic', 'center', 'centre'
        ]
        
        # Check if the candidate contains non-name indicators
        lower_name = name.lower()
        if any(indicator in lower_name for indicator in non_name_indicators):
            return False
        
        # Names typically don't have numbers
        if re.search(r'\d', name):
            return False
        
        # Must pass all validation
        return True
    
    def _extract_interpreting_provider(self) -> ExtractedField:
        """Extract interpreting provider (radiologist) from the text"""
        # Look specifically for "Electronically Signed: Dr. X" pattern - very common in reports
        signed_patterns = [
            r"Electronically\s+Signed:?\s*(?:Dr\.?\s*)?([A-Za-z\s\-'.]+),?\s*(?:MD|FRCPC|DO|PhD)?",
            r"Electronically\s+Signed:?\s*(?:by\s*)?(?:Dr\.?\s*)?([A-Za-z\s\-'.]+),?\s*(?:MD|FRCPC|DO|PhD)?",
            r"Signed:?\s*(?:Dr\.?\s*)?([A-Za-z\s\-'.]+),?\s*(?:MD|FRCPC|DO|PhD)?"
        ]
        
        for pattern in signed_patterns:
            match = re.search(pattern, self.preprocessed_text, re.IGNORECASE)
            if match:
                provider = match.group(1).strip()
                if provider and len(provider) > 2:  # Avoid very short matches
                    # Clean up standard prefixes/suffixes
                    provider = re.sub(r'^(?:Dr\.?|MD|Doctor)\s+', '', provider, flags=re.IGNORECASE)
                    provider = re.sub(r'\s+(?:MD|Ph\.?D|DO|DDS|FRCPC)$', '', provider, flags=re.IGNORECASE)
                    self.metadata['extraction_stats']['interpreting_provider'] = "Found via electronically signed pattern"
                    return ExtractedField(provider, 0.95, "signed_pattern")
        
        # Common patterns for interpreting provider
        patterns = [
            r"(?:Interpreted|INTERPRETED|Read|READ)\s+(?:by|BY)[:;]?\s*(?:Dr\.?\s*)?([A-Za-z\s\-'.]+)",
            r"(?:Radiologist|RADIOLOGIST|Interpreter|INTERPRETER)[:;]?\s*(?:Dr\.?\s*)?([A-Za-z\s\-'.]+)",
            r"(?:Report\s+by|REPORT\s+BY)[:;]?\s*(?:Dr\.?\s*)?([A-Za-z\s\-'.]+)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, self.preprocessed_text, re.IGNORECASE)
            if match:
                provider = match.group(1).strip()
                if provider and len(provider) > 2:  # Avoid very short matches
                    # Clean up standard prefixes/suffixes
                    provider = re.sub(r'^(?:Dr\.?|MD|Doctor)\s+', '', provider, flags=re.IGNORECASE)
                    provider = re.sub(r'\s+(?:MD|Ph\.?D|DO|DDS|FRCPC)$', '', provider, flags=re.IGNORECASE)
                    self.metadata['extraction_stats']['interpreting_provider'] = "Found via standard pattern"
                    return ExtractedField(provider, 0.9, "standard_pattern")
        
        # Look for Dr. Name pattern in last part of document (common signature location)
        doc_end = self.preprocessed_text[-min(500, len(self.preprocessed_text)):]
        doctor_pattern = r"(?:Dr\.?\s+)([A-Z][a-z]+\s+[A-Z][a-z]+)"
        match = re.search(doctor_pattern, doc_end)
        if match:
            provider = match.group(1).strip()
            if provider and len(provider) > 4:  # More strict for this pattern
                self.metadata['extraction_stats']['interpreting_provider'] = "Found via Dr. pattern at end"
                return ExtractedField(provider, 0.85, "doctor_pattern")
        
        # Search for a provider name near a date at the end of the document
        provider_date_pattern = r"([A-Za-z\s\-'.]{2,30})\s+\d{1,2}/\d{1,2}/\d{4}"
        match = re.search(provider_date_pattern, doc_end)
        if match:
            provider = match.group(1).strip()
            if provider and len(provider) > 4 and not any(common in provider.lower() for common in ['page', 'printed', 'date']):
                self.metadata['extraction_stats']['interpreting_provider'] = "Found via provider near date"
                return ExtractedField(provider, 0.7, "date_proximity")
        
        return ExtractedField("N/A", 0.0, "not_found")
    
    def _get_section_boundary_regex(self) -> str:
        """
        Returns a regex pattern for identifying section boundaries in the document.
        Supports both English and French section headers.
        """
        if self.language == "fr":
            # French section headers
            sections = [
                "RENSEIGNEMENTS CLINIQUES", "HISTOIRE CLINIQUE", "ANTÉCÉDENTS", "HISTORIQUE",
                "TECHNIQUE", "PROTOCOLE", "PROCÉDURE",
                "RÉSULTATS", "OBSERVATIONS", "CONSTATATIONS", "TROUVAILLES",
                "IMPRESSION", "CONCLUSION", "INTERPRÉTATION", "OPINION",
                "RECOMMANDATION", "SUIVI", "CONDUITE À TENIR",
                "BIRADS", "CATÉGORIE", "CLASSIFICATION",
                "MÉDECIN TRAITANT", "MÉDECIN RÉFÉRENT", "RÉFÉRÉ PAR",
                "RADIOLOGUE", "INTERPRÉTÉ PAR", "SIGNÉ PAR"
            ]
        else:
            # English section headers
            sections = [
                "CLINICAL INFORMATION", "HISTORY", "CLINICAL HISTORY", "INDICATION",
                "TECHNIQUE", "PROCEDURE", "PROTOCOL",
                "FINDINGS", "RESULT", "OBSERVATION",
                "IMPRESSION", "CONCLUSION", "INTERPRETATION", "ASSESSMENT",
                "RECOMMENDATION", "FOLLOW-UP", "PLAN",
                "BIRADS", "CATEGORY", "CLASSIFICATION",
                "REFERRING PHYSICIAN", "REFERRING DOCTOR", "REFERRED BY",
                "RADIOLOGIST", "INTERPRETED BY", "SIGNED BY"
            ]
        
        # Create a regex pattern that matches any of these section headers
        # followed by a colon or line break
        pattern = r"(?:^|\n)(?:" + "|".join(sections) + r")(?::|;|\s*\n)"
        return pattern
    
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
        
        # 1. Ensure consistency between findings and impression
        findings = get_safe_value("findings")
        impression = get_safe_value("impression")
        
        # If findings is N/A but impression is available
        if findings == "N/A" and impression != "N/A":
            # Use impression for findings
            self.extracted_fields["findings"] = ExtractedField(impression, 0.7, "derived_from_impression")
            self.metadata['extraction_stats']['findings'] = "Derived from impression"
        
        # If impression is N/A but findings is available
        if impression == "N/A" and findings != "N/A":
            # Try to extract the last paragraph from findings as impression
            paragraphs = findings.split('\n\n')
            if len(paragraphs) > 1:
                potential_impression = paragraphs[-1]
                if len(potential_impression) > 20:  # Reasonable length for an impression
                    self.extracted_fields["impression"] = ExtractedField(potential_impression, 0.7, "derived_from_findings")
                    self.metadata['extraction_stats']['impression'] = "Derived from findings (last paragraph)"
            else:
                # If single paragraph, use the whole thing
                self.extracted_fields["impression"] = ExtractedField(findings, 0.7, "derived_from_findings")
                self.metadata['extraction_stats']['impression'] = "Derived from findings (same content)"
        
        # 2. Ensure clinical_history and patient_history are synchronized
        clinical_history = get_safe_value("clinical_history")
        patient_history = get_safe_value("patient_history")
        
        # If clinical_history is empty but patient_history is available
        if clinical_history == "N/A" and patient_history != "N/A":
            self.extracted_fields["clinical_history"] = ExtractedField(patient_history, 0.7, "derived_from_patient_history")
            self.metadata['extraction_stats']['patient_history'] = "Derived from clinical_history"
        
        # If patient_history is empty but clinical_history is available
        if patient_history == "N/A" and clinical_history != "N/A":
            self.extracted_fields["patient_history"] = ExtractedField(clinical_history, 0.7, "derived_from_clinical_history")
            self.metadata['extraction_stats']['patient_history'] = "Derived from clinical_history"
        
        # 3. Ensure exam_date has a reasonable format
        try:
            exam_date = get_safe_value("exam_date")
            if exam_date != "N/A":
                # Normalize date format if possible
                normalized_date = self._normalize_date_format(exam_date)
                if normalized_date != exam_date:
                    self.extracted_fields["exam_date"] = ExtractedField(normalized_date, 0.8, "normalized")
                    self.metadata['extraction_stats']['exam_date'] = "Date format normalized"
        except Exception as e:
            self.metadata['validation_issues'].append(f"Error extracting exam_date: {str(e)}")
            
        # 4. Ensure consistency between findings and mammograph_results
        # If mammograph_results is in expected fields, sync with findings
        if "mammograph_results" in self.EXPECTED_FIELDS:
            mammograph_results = get_safe_value("mammograph_results")
            if mammograph_results == "N/A" and findings != "N/A":
                self.extracted_fields["mammograph_results"] = ExtractedField(findings, 0.7, "derived_from_findings")
                self.metadata['extraction_stats']['mammograph_results'] = "Derived from findings"
        
        # 5. Record validation results
        for field in self.EXPECTED_FIELDS:
            value = get_safe_value(field)
            if value == "N/A":
                self.metadata['validation_issues'].append(f"Field {field} not found")
    
    def _normalize_date_format(self, date_str: str) -> str:
        """
        Normalize date format to YYYY-MM-DD.
        
        Args:
            date_str: Date string in various formats
            
        Returns:
            Normalized date string if possible, otherwise original string
        """
        # Try to identify common formats and convert
        
        # Format: MM/DD/YYYY or DD/MM/YYYY
        match = re.match(r'(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})', date_str)
        if match:
            m, d, y = match.groups()
            # Assume MM/DD/YYYY format
            return f"{y}-{m.zfill(2)}-{d.zfill(2)}"
        
        # Format: YYYY/MM/DD
        match = re.match(r'(\d{4})[/.-](\d{1,2})[/.-](\d{1,2})', date_str)
        if match:
            y, m, d = match.groups()
            return f"{y}-{m.zfill(2)}-{d.zfill(2)}"
        
        # Format: Month DD, YYYY
        match = re.match(r'([A-Za-z]+)\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})', date_str)
        if match:
            month_name, d, y = match.groups()
            month_map = {
                'jan': '01', 'feb': '02', 'mar': '03', 'apr': '04', 
                'may': '05', 'jun': '06', 'jul': '07', 'aug': '08',
                'sep': '09', 'oct': '10', 'nov': '11', 'dec': '12'
            }
            month_abbr = month_name.lower()[:3]
            if month_abbr in month_map:
                return f"{y}-{month_map[month_abbr]}-{d.zfill(2)}"
        
        # If no pattern matches, return original
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

    def _detect_language(self, text: str) -> str:
        """
        Detect the primary language of the document (English or French).
        
        Args:
            text: The preprocessed text
            
        Returns:
            Language code ('en' or 'fr')
        """
        # Define language indicators
        french_indicators = [
            'examen', 'résultat', 'médicale', 'clinique', 'mammographie', 
            'rapport', 'patiente', 'médecin', 'requête', 'imagerie',
            'protocole', 'sein', 'signé', 'transcrit', 'date d\'examen'
        ]
        
        english_indicators = [
            'examination', 'result', 'medical', 'clinic', 'mammography',
            'report', 'patient', 'doctor', 'requisition', 'imaging',
            'protocol', 'breast', 'signed', 'transcribed', 'exam date'
        ]
        
        # Check for presence of indicators in the text (case insensitive)
        normalized_text = text.lower()
        
        french_count = sum(1 for word in french_indicators if word in normalized_text)
        english_count = sum(1 for word in english_indicators if word in normalized_text)
        
        # Add extra weight to certain strong indicators
        if 'médical' in normalized_text or 'clinique' in normalized_text:
            french_count += 2
        if 'medical' in normalized_text or 'clinical' in normalized_text:
            english_count += 2
        
        # Determine language based on indicator counts
        return 'fr' if french_count > english_count else 'en'

    def enhance_with_llm(self, llm_client=None, confidence_threshold=0.7) -> Dict[str, ExtractedField]:
        """
        Enhance extraction results using an LLM for fields with low confidence.
        
        Args:
            llm_client: The LLM client to use for enhancement (if None, will return original fields)
            confidence_threshold: Fields with confidence below this threshold will be enhanced
            
        Returns:
            Dictionary of enhanced extracted fields
        """
        # If no LLM client is provided, return the original fields
        if llm_client is None:
            return self.extracted_fields
        
        # Create a copy of the extracted fields to avoid modifying the original
        enhanced_fields = self.extracted_fields.copy()
        
        # Identify fields that need enhancement
        fields_to_enhance = []
        for field_name in self.EXPECTED_FIELDS:
            if field_name in enhanced_fields:
                field = enhanced_fields[field_name]
                if field.confidence < confidence_threshold and field.value != "N/A":
                    fields_to_enhance.append(field_name)
                elif field.value == "N/A":
                    fields_to_enhance.append(field_name)
        
        if not fields_to_enhance:
            return enhanced_fields  # No fields need enhancement
        
        # Prepare the prompt for the LLM
        prompt = self._prepare_llm_prompt(fields_to_enhance)
        
        try:
            # Call the LLM
            llm_response = llm_client.generate_text(prompt)
            
            # Parse the LLM response and update the fields
            enhanced_fields = self._parse_llm_response(llm_response, enhanced_fields, fields_to_enhance)
            
            # Record that LLM enhancement was used
            self.metadata['llm_enhancement'] = {
                'fields_enhanced': fields_to_enhance,
                'prompt_length': len(prompt),
                'response_length': len(llm_response) if llm_response else 0
            }
        except Exception as e:
            self.metadata['validation_issues'].append(f"LLM enhancement failed: {str(e)}")
        
        return enhanced_fields

    def _prepare_llm_prompt(self, fields_to_enhance: List[str]) -> str:
        """
        Prepare a prompt for the LLM to enhance extraction.
        
        Args:
            fields_to_enhance: List of field names to enhance
            
        Returns:
            Prompt string for the LLM
        """
        # Start with a system instruction based on language
        if self.language == 'fr':
            prompt = f"""Vous êtes un expert en analyse de documents médicaux. J'ai besoin de votre aide pour extraire des informations spécifiques d'un rapport médical en français.
Le rapport est un {self._get_exam_type_description()} et contient des informations importantes sur le patient et les résultats de l'examen.

Voici le texte du rapport:
```
{self.preprocessed_text[:2000]}  # Limité aux 2000 premiers caractères pour éviter les limites de tokens
```

Veuillez extraire les informations suivantes de manière précise:
"""
        else:
            prompt = f"""You are an expert medical document analyzer. I need your help extracting specific information from a medical report.
The report is a {self._get_exam_type_description()} in English.

Here is the text of the report:
```
{self.preprocessed_text[:2000]}  # Limit to first 2000 chars to avoid token limits
```

Please extract the following information:
"""
        
        # Add instructions for each field to enhance, with language-specific phrasing
        for field_name in fields_to_enhance:
            if self.language == 'fr':
                prompt += f"\n- {field_name.replace('_', ' ').title()}: "
                if field_name == "clinical_history":
                    prompt += "Extrayez l'historique clinique du patient ou la raison de l'examen."
                elif field_name == "findings":
                    prompt += "Extrayez les résultats détaillés ou les observations de l'examen."
                elif field_name == "impression":
                    prompt += "Extrayez l'impression ou la conclusion du radiologue."
                elif field_name == "recommendation":
                    prompt += "Extrayez les recommandations pour le suivi ou les examens supplémentaires."
                elif field_name == "birads_score":
                    prompt += "Extrayez le score BI-RADS (un chiffre de 0 à 6 ou avec des lettres comme 4A, 4B, 4C)."
                elif field_name == "exam_date":
                    prompt += "Extrayez la date à laquelle l'examen a été effectué (au format AAAA-MM-JJ si possible)."
                elif field_name == "facility":
                    prompt += "Extrayez le nom de l'établissement médical où l'examen a été effectué."
                elif field_name == "exam_type":
                    prompt += "Extrayez le type d'examen effectué (par exemple, mammographie, échographie, IRM)."
                elif field_name == "referring_provider":
                    prompt += "Extrayez le nom du médecin référent ou du prescripteur."
                elif field_name == "interpreting_provider":
                    prompt += "Extrayez le nom du radiologue ou du médecin qui a interprété les résultats."
            else:
                prompt += f"\n- {field_name.replace('_', ' ').title()}: "
                if field_name == "clinical_history":
                    prompt += "Extract the patient's clinical history or reason for the examination."
                elif field_name == "findings":
                    prompt += "Extract the detailed findings or results of the examination."
                elif field_name == "impression":
                    prompt += "Extract the radiologist's impression or conclusion."
                elif field_name == "recommendation":
                    prompt += "Extract any recommendations for follow-up or further testing."
                elif field_name == "birads_score":
                    prompt += "Extract the BI-RADS score (a number from 0-6 or with letters like 4A, 4B, 4C)."
                elif field_name == "exam_date":
                    prompt += "Extract the date when the examination was performed (in YYYY-MM-DD format if possible)."
                elif field_name == "facility":
                    prompt += "Extract the name of the medical facility where the examination was performed."
                elif field_name == "exam_type":
                    prompt += "Extract the type of examination performed (e.g., mammogram, ultrasound, MRI)."
                elif field_name == "referring_provider":
                    prompt += "Extract the name of the referring physician or provider."
                elif field_name == "interpreting_provider":
                    prompt += "Extract the name of the radiologist or provider who interpreted the results."
        
        # Add formatting instructions with language-specific phrasing
        if self.language == 'fr':
            prompt += """

Pour chaque champ, fournissez uniquement l'information extraite sans commentaire supplémentaire.
Si vous ne trouvez pas d'information pour un champ, répondez 'N/A' pour ce champ.
Formatez votre réponse comme suit:
Nom du Champ: Information Extraite

Merci!"""
        else:
            prompt += """

For each field, provide only the extracted information without any additional commentary.
If you cannot find information for a field, respond with 'N/A' for that field.
Format your response as:
Field Name: Extracted Information

Thank you!"""
        
        return prompt

    def _parse_llm_response(self, llm_response: str, current_fields: Dict[str, ExtractedField], 
                            fields_to_enhance: List[str]) -> Dict[str, ExtractedField]:
        """
        Parse the LLM response and update the extracted fields.
        
        Args:
            llm_response: The response from the LLM
            current_fields: The current extracted fields
            fields_to_enhance: List of field names that were requested for enhancement
            
        Returns:
            Updated dictionary of extracted fields
        """
        if not llm_response:
            return current_fields
        
        # Create a copy to avoid modifying the original
        enhanced_fields = current_fields.copy()
        
        # Process each line of the response
        lines = llm_response.strip().split('\n')
        for line in lines:
            # Skip empty lines
            if not line.strip():
                continue
            
            # Try to parse the line as "Field Name: Value"
            parts = line.split(':', 1)
            if len(parts) != 2:
                continue
            
            field_name = parts[0].strip().lower().replace(' ', '_')
            field_value = parts[1].strip()
            
            # Check if this is a field we want to enhance
            if field_name in fields_to_enhance or any(f.lower().replace(' ', '_') == field_name for f in fields_to_enhance):
                # Map common variations to our standard field names
                if field_name == "clinical_history" or field_name == "patient_history":
                    field_name = "clinical_history"
                elif field_name == "results" or field_name == "observations":
                    field_name = "findings"
                elif field_name == "conclusion" or field_name == "assessment":
                    field_name = "impression"
                elif field_name == "follow_up" or field_name == "plan":
                    field_name = "recommendation"
                elif field_name == "birads" or field_name == "category":
                    field_name = "birads_score"
                elif field_name == "referring_doctor" or field_name == "referring_physician":
                    field_name = "referring_provider"
                elif field_name == "radiologist" or field_name == "interpreter":
                    field_name = "interpreting_provider"
                
                # Skip if the field is not in our expected fields
                if field_name not in self.EXPECTED_FIELDS:
                    continue
                
                # Skip if the value is empty or N/A
                if not field_value or field_value.upper() == "N/A":
                    continue
                
                # Get the current field
                current_field = enhanced_fields.get(field_name)
                current_value = current_field.value if current_field else "N/A"
                current_confidence = current_field.confidence if current_field else 0.0
                
                # Decide whether to use the LLM value
                if current_value == "N/A" or current_confidence < 0.7:
                    enhanced_fields[field_name] = ExtractedField(field_value, 0.8, "llm")
                elif len(field_value) > len(current_value) * 1.5:
                    # LLM found significantly more content
                    enhanced_fields[field_name] = ExtractedField(field_value, 0.85, "llm_expanded")
                elif current_confidence < 0.8 and len(field_value) > 10:
                    # Current confidence is moderate and LLM found reasonable content
                    enhanced_fields[field_name] = ExtractedField(field_value, 0.8, "llm_alternative")
        
        return enhanced_fields

    def _get_exam_type_description(self) -> str:
        """Get a description of the exam type for the LLM prompt"""
        exam_type = self.extracted_fields.get("exam_type")
        if exam_type and hasattr(exam_type, 'value') and exam_type.value != "N/A":
            return exam_type.value
        
        # Try to infer from the text
        if "mammogram" in self.preprocessed_text.lower() or "mammographie" in self.preprocessed_text.lower():
            return "mammogram"
        elif "ultrasound" in self.preprocessed_text.lower() or "échographie" in self.preprocessed_text.lower():
            return "breast ultrasound"
        elif "mri" in self.preprocessed_text.lower() or "irm" in self.preprocessed_text.lower():
            return "breast MRI"
        else:
            return "breast imaging examination"


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