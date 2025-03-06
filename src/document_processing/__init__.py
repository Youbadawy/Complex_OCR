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

__all__ = [
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
    'PDF2IMAGE_AVAILABLE'
]

# Initialize document_processing package 