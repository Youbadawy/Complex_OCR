"""
Document processing module for Medical Report Processor application.
This package contains OCR and text analysis functionality.
"""

from .ocr import (
    process_pdf, process_pdf_batch, debug_ocr_process,
    PYMUPDF_AVAILABLE, PADDLE_AVAILABLE, PDF2IMAGE_AVAILABLE
)
from .text_analysis import (
    process_document_text, extract_structured_data, extract_birads_score,
    extract_provider_info, extract_signed_by
)
from .pdf_processor import (
    extract_sections, extract_patient_info,
    extract_date_from_text, extract_exam_type
)

# Import new modules
try:
    from .extraction_pipeline import (
        ExtractionPipeline, extract_all_fields_from_text, ALL_MODULES_AVAILABLE
    )
    from .multilingual_extraction import (
        detect_language, extract_with_language_specific_patterns
    )
    from .report_validation import (
        validate_report_data, extract_and_validate_report,
        validate_and_enhance_extraction, cross_validate_extractions
    )
    from .llm_extraction import (
        extract_with_llm, enhance_extraction_with_llm, LLM_AVAILABLE
    )
    
    ENHANCED_PIPELINE_AVAILABLE = True
except ImportError as e:
    import logging
    logging.warning(f"Enhanced extraction pipeline not fully available: {str(e)}")
    ENHANCED_PIPELINE_AVAILABLE = False

__all__ = [
    # Core OCR and processing
    'process_pdf',
    'process_pdf_batch',
    'debug_ocr_process',
    'process_document_text',
    'extract_structured_data',
    'extract_birads_score',
    'extract_provider_info',
    'extract_signed_by',
    'extract_sections',
    'extract_patient_info',
    'extract_date_from_text', 
    'extract_exam_type',
    'PYMUPDF_AVAILABLE',
    'PADDLE_AVAILABLE',
    'PDF2IMAGE_AVAILABLE',
    
    # Enhanced extraction pipeline
    'ExtractionPipeline',
    'extract_all_fields_from_text',
    'detect_language',
    'extract_with_language_specific_patterns',
    'validate_report_data',
    'extract_and_validate_report',
    'validate_and_enhance_extraction',
    'cross_validate_extractions',
    'extract_with_llm',
    'enhance_extraction_with_llm',
    'LLM_AVAILABLE',
    'ENHANCED_PIPELINE_AVAILABLE',
    'ALL_MODULES_AVAILABLE'
]

# Initialize document_processing package 