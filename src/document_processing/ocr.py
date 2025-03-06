"""
OCR and document processing functionality for Medical Report Processor application.
"""
import os
import io
import tempfile
import logging
import numpy as np
import pandas as pd
import streamlit as st
from io import BytesIO
from typing import Dict, List, Any, BinaryIO, Optional, Union
import re
import hashlib
from PIL import Image
import json
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
import torch

# Set up logging
logger = logging.getLogger(__name__)

# Try importing OCR dependencies
try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False
    logger.warning("PaddleOCR not available. OCR features will be limited.")

# Try importing pytesseract
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
    
    # Configure Tesseract path based on OS
    if os.name == 'nt':  # Windows
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    else:
        pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'  # Linux/Mac path
except ImportError:
    TESSERACT_AVAILABLE = False
    logger.warning("Tesseract not available. OCR features will be limited.")

# Try importing pdfplumber
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    logger.warning("pdfplumber not available. Text extraction will be limited.")

# Try importing PyMuPDF for PDF processing
PYMUPDF_AVAILABLE = False
try:
    # Create static directory if it doesn't exist (needed by PyMuPDF)
    if not os.path.exists('static'):
        os.makedirs('static')
        logger.info("Created 'static/' directory for PyMuPDF")
    
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except (ImportError, RuntimeError) as e:
    logger.warning(f"PyMuPDF not available or configuration issue: {e}. PDF processing will be limited.")
    
# Try importing alternative PDF libraries
PDF2IMAGE_AVAILABLE = False
try:
    from pdf2image import convert_from_bytes
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    logger.warning("pdf2image not available. PDF to image conversion will be unavailable.")

# Try importing Donut document understanding model
try:
    from transformers import DonutProcessor, VisionEncoderDecoderModel
    DONUT_AVAILABLE = True
except ImportError:
    DONUT_AVAILABLE = False
    logger.warning("Transformers library not available. Donut document understanding will be unavailable.")

@st.cache_resource(show_spinner="Initializing OCR Engine...")
def init_paddle():
    """
    Initialize the PaddleOCR engine
    
    Returns:
        PaddleOCR: Initialized OCR engine or None if unavailable
    """
    if not PADDLE_AVAILABLE:
        logger.warning("PaddleOCR not installed. OCR functionality will be limited.")
        return None
    
    try:
        # Initialize PaddleOCR with English and rotation detection
        logger.info("Initializing PaddleOCR with English language support")
        ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=True)
        logger.info("PaddleOCR initialization complete")
        return ocr
    except Exception as e:
        logger.error(f"Error initializing PaddleOCR: {e}")
        return None

@st.cache_resource
def init_donut():
    """
    Initialize Donut document understanding model
    
    Returns:
        tuple: (processor, model) or (None, None) if unavailable
    """
    try:
        import torch
        from transformers import DonutProcessor, VisionEncoderDecoderModel
        
        processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
        model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
        
        # Move to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        return processor, model
    except Exception as e:
        logger.error(f"Error initializing Donut model: {e}")
        return None, None

def simple_ocr(img_array):
    """
    Run simple OCR using pytesseract
    
    Args:
        img_array: numpy array of image
        
    Returns:
        str: Extracted text
    """
    if not TESSERACT_AVAILABLE:
        return ""
        
    try:
        # Run OCR
        text = pytesseract.image_to_string(img_array)
        return text
    except Exception as e:
        logger.error(f"Error in simple OCR: {e}")
        return ""

def hybrid_ocr(img):
    """
    Run OCR using multiple engines and select best result
    
    Args:
        img: PIL Image or numpy array
        
    Returns:
        str: Best extracted text
    """
    results = []
    
    # Convert PIL Image to numpy array if needed
    if hasattr(img, 'convert'):  # PIL Image
        img_array = np.array(img.convert('RGB'))
    else:  # Assume numpy array
        img_array = img
    
    # Try PaddleOCR
    ocr_engine = init_paddle()
    if ocr_engine:
        try:
            paddle_result = ocr_engine.ocr(img_array, cls=True)
            
            # Extract text from PaddleOCR result
            paddle_text = ""
            if isinstance(paddle_result, list):
                for page_result in paddle_result:
                    if page_result and isinstance(page_result, list):
                        for line in page_result:
                            if line and isinstance(line, list) and len(line) == 2:
                                text, confidence = line[1]
                                paddle_text += text + " "
            
            if paddle_text.strip():
                results.append(paddle_text)
                logger.debug(f"PaddleOCR extracted {len(paddle_text)} chars")
        except Exception as e:
            logger.error(f"PaddleOCR error: {e}")
    
    # Try Tesseract
    if TESSERACT_AVAILABLE:
        try:
            tesseract_text = simple_ocr(img_array)
            if tesseract_text.strip():
                results.append(tesseract_text)
                logger.debug(f"Tesseract extracted {len(tesseract_text)} chars")
        except Exception as e:
            logger.error(f"Tesseract error: {e}")
    
    # Select best result (longest text for now - this is a simple heuristic)
    if results:
        best_text = max(results, key=len)
        logger.info(f"Selected best OCR result with {len(best_text)} chars")
        return best_text
    
    logger.warning("No OCR results available")
    return ""

def is_deidentified_document(text):
    """
    Check if a document appears to be deidentified
    
    Args:
        text: Document text
        
    Returns:
        bool: True if deidentified
    """
    deidentified_patterns = [
        r'\b(PROTECTED\s+[A-Z]|REDACTED|\[REDACTED\]|CONFIDENTIAL)\b',
        r'\b(XXXX|____|\*\*\*\*|deidentified)\b'
    ]
    
    for pattern in deidentified_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    
    return False

def process_pdf(uploaded_file):
    """
    Process a PDF file to extract text, images, and structured data
    
    Args:
        uploaded_file: Streamlit uploaded file object or path to a PDF file
        
    Returns:
        dict: Extracted data including text, images, and structured information
    """
    if uploaded_file is None:
        return None
    
    try:
        # Create a file-like object for both string paths and uploaded files
        file_wrapper = None
        
        # Handle different input types
        if isinstance(uploaded_file, str):
            # Input is a file path
            if not os.path.exists(uploaded_file):
                raise FileNotFoundError(f"File not found: {uploaded_file}")
                
            # Create a file-like object
            class FileWrapper:
                def __init__(self, path):
                    self.path = path
                    self.name = os.path.basename(path)
                    self._file = open(path, 'rb')
                    self._content = None
                    
                def read(self):
                    self._file.seek(0)
                    if self._content is None:
                        self._content = self._file.read()
                    return self._content
                    
                def seek(self, pos):
                    self._file.seek(pos)
                    
                def getvalue(self):
                    if self._content is None:
                        self._content = self.read()
                    return self._content
                    
                def close(self):
                    self._file.close()
                    
                def __enter__(self):
                    return self
                    
                def __exit__(self, exc_type, exc_val, exc_tb):
                    self.close()
            
            # Create our wrapper
            file_wrapper = FileWrapper(uploaded_file)
            file_content = file_wrapper.read()
            filename = file_wrapper.name
        else:
            # Input is a Streamlit UploadedFile
            file_wrapper = uploaded_file
            file_content = file_wrapper.read()
            filename = file_wrapper.name
        
        # Generate MD5 hash
        md5_hash = hashlib.md5(file_content).hexdigest()
        
        # Reset file pointer
        file_wrapper.seek(0)
        
        # Initialize result dictionary
        result = {
            'filename': filename,
            'md5_hash': md5_hash,
            'file_size': len(file_content),
            'success': False
        }
        
        # Extract text from all pages using multiple methods
        all_text = []
        extracted_images = []
        
        # First attempt: Use pdfplumber (preferred for text extraction)
        if PDFPLUMBER_AVAILABLE:
            try:
                with pdfplumber.open(file_wrapper) as pdf:
                    # Get basic document info
                    result['page_count'] = len(pdf.pages)
                    
                    # Process all pages and combine text
                    for i, page in enumerate(pdf.pages):
                        # Extract text
                        page_text = page.extract_text()
                        
                        # If meaningful text was extracted
                        if page_text and len(page_text.strip()) > 50:
                            all_text.append(page_text)
                            logger.info(f"Extracted text from page {i+1} using pdfplumber")
                        else:
                            # Fallback to OCR if text extraction fails or returns minimal text
                            logger.info(f"Using OCR fallback for page {i+1}")
                            try:
                                # Convert page to image
                                img = page.to_image(resolution=300).original
                                
                                # Add to extracted images
                                img_byte_arr = BytesIO()
                                img.save(img_byte_arr, format='PNG')
                                extracted_images.append(img_byte_arr.getvalue())
                                
                                # Use hybrid OCR
                                ocr_text = hybrid_ocr(img)
                                all_text.append(ocr_text)
                                logger.info(f"Extracted {len(ocr_text)} chars with OCR fallback on page {i+1}")
                            except Exception as ocr_error:
                                logger.error(f"OCR fallback failed on page {i+1}: {ocr_error}")
                                all_text.append("")  # Add empty text as placeholder
                
                # If we have both text and images, mark this method as successful
                if all_text:
                    result['success'] = True
                    logger.info(f"Successfully processed {filename} with pdfplumber + OCR fallback")
            except Exception as e:
                logger.warning(f"pdfplumber extraction failed: {e}, falling back to PyMuPDF")
        
        # Second attempt: Use PyMuPDF if pdfplumber failed
        if not result.get('success', False) and PYMUPDF_AVAILABLE:
            try:
                # Reset file pointer
                file_wrapper.seek(0)
                
                with fitz.open(stream=file_content, filetype="pdf") as pdf_document:
                    # Get basic document info
                    result['page_count'] = len(pdf_document)
                    
                    # Extract text and images from all pages
                    all_text = []
                    extracted_images = []
                    
                    for i, page in enumerate(pdf_document):
                        # Try to extract text
                        text = page.get_text()
                        
                        # If meaningful text was extracted
                        if text and len(text.strip()) > 50:
                            all_text.append(text)
                            logger.info(f"Extracted text from page {i+1} using PyMuPDF")
                        else:
                            # Fallback to OCR
                            logger.info(f"Using OCR fallback for page {i+1}")
                            try:
                                # Render page to image
                                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                                img_bytes = pix.tobytes("png")
                                extracted_images.append(img_bytes)
                                
                                # Convert to numpy array for OCR
                                img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
                                
                                # Use hybrid OCR
                                ocr_text = hybrid_ocr(img_array)
                                all_text.append(ocr_text)
                                logger.info(f"Extracted {len(ocr_text)} chars with OCR fallback on page {i+1}")
                            except Exception as ocr_error:
                                logger.error(f"OCR fallback failed on page {i+1}: {ocr_error}")
                                all_text.append("")  # Add empty text as placeholder
                    
                    # Set success flag
                    result['success'] = True
                    logger.info(f"Successfully processed {filename} with PyMuPDF + OCR fallback")
            except Exception as e:
                logger.warning(f"PyMuPDF extraction failed: {e}, falling back to pure OCR")
        
        # Third attempt: Full OCR if other methods failed
        if not result.get('success', False) and PDF2IMAGE_AVAILABLE:
            try:
                # Reset file pointer
                file_wrapper.seek(0)
                
                # Use pdf2image to convert pages
                images = convert_from_bytes(file_content)
                logger.info(f"Converted PDF to {len(images)} images for OCR processing")
                
                # Process each page with OCR
                all_text = []
                extracted_images = []
                
                for i, img in enumerate(images):
                    # Store image for display
                    img_byte_arr = BytesIO()
                    img.save(img_byte_arr, format='PNG')
                    extracted_images.append(img_byte_arr.getvalue())
                    
                    # Process with hybrid OCR
                    ocr_text = hybrid_ocr(img)
                    all_text.append(ocr_text)
                    logger.info(f"Extracted {len(ocr_text)} chars with OCR on page {i+1}")
                
                # Set success flag
                result['success'] = True
                logger.info(f"Successfully processed {filename} with full OCR")
            except Exception as e:
                logger.error(f"Full OCR extraction failed: {e}")
        
        # Store extracted images in the result
        result['images'] = extracted_images
        
        # Combine all text into a single document
        combined_text = "\n\n".join(all_text)
        logger.info(f"Combined text length: {len(combined_text)} characters")
        result['text'] = combined_text
        result['raw_ocr_text'] = combined_text
        
        # Check if we actually extracted any text
        if not combined_text.strip():
            logger.warning("No text extracted from document")
            result['warning'] = "No text could be extracted from the document"
            
            # Create minimal structured data
            result['structured_data'] = {
                'patient_name': "Not Available",
                'exam_date': "Not Available",
                'exam_type': "Not Available",
                'birads_score': "Not Available",
                'raw_ocr_text': "No text extracted from document"
            }
            
            # Add minimal metadata
            result['metadata'] = {
                "md5_hash": md5_hash,
                "filename": filename,
                "pages": result.get('page_count', 0),
                "processing_date": pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                "is_deidentified": False,
                "extraction_failed": True
            }
            
            return result.get('structured_data', {}), result.get('metadata', {})
        
        # Process the combined text to extract structured data
        try:
            # Import text analysis functions
            from document_processing.text_analysis import process_document_text
            structured_data = process_document_text(combined_text)
            structured_data['raw_ocr_text'] = combined_text
            result['structured_data'] = structured_data
            
            # Process the extracted text to get structured data
            try:
                # Import text analysis functions
                from document_processing.text_analysis import validate_field_types
                structured_data = validate_field_types(structured_data)
            except Exception as e:
                logger.error(f"Field validation error: {str(e)}")
        except Exception as extraction_error:
            logger.error(f"Structured data extraction error: {extraction_error}")
            
            # Create basic structured data
            result['structured_data'] = {
                'patient_name': "Not Available",
                'exam_date': "Not Available",
                'exam_type': "Not Available",
                'birads_score': "Not Available",
                'raw_ocr_text': combined_text
            }
        
        # Extract metadata
        try:
            metadata = {
                "md5_hash": md5_hash,
                "filename": filename,
                "pages": result.get('page_count', len(all_text)),
                "processing_date": pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                "is_deidentified": is_deidentified_document(combined_text)
            }
            
            # Add key metadata from structured data
            if 'structured_data' in result and result['structured_data']:
                if 'exam_type' in result['structured_data']:
                    metadata['exam_type'] = result['structured_data']['exam_type']
                if 'exam_date' in result['structured_data']:
                    metadata['exam_date'] = result['structured_data']['exam_date']
                if 'birads_score' in result['structured_data']:
                    metadata['birads_score'] = result['structured_data']['birads_score']
            
            result['metadata'] = metadata
            
        except Exception as metadata_error:
            logger.error(f"Error extracting metadata: {metadata_error}")
            result['metadata'] = {
                "md5_hash": md5_hash,
                "filename": filename,
                "processing_date": pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        
        # Add findings field for database compatibility
        if 'structured_data' in result and result['structured_data']:
            result['findings'] = {
                'birads_score': result['structured_data'].get('birads_score', None),
                'exam_type': result['structured_data'].get('exam_type', None),
                'exam_date': result['structured_data'].get('exam_date', None),
                'patient_name': result['structured_data'].get('patient_name', None),
                'impression': result['structured_data'].get('impression', None),
                'testing_provider': result['structured_data'].get('testing_provider', None)
            }
        else:
            result['findings'] = {}
        
        # Try to save to database if available
        try:
            from database.operations import save_to_db
            save_to_db({
                "md5_hash": md5_hash,
                "filename": filename,
                "patient_name": result['structured_data'].get('patient_name', 'Not Available'),
                "exam_date": result['structured_data'].get('exam_date', 'Not Available'),
                "findings": result['structured_data'],
                "metadata": result['metadata'],
                "raw_ocr_text": combined_text
            })
        except Exception as db_error:
            logger.error(f"Database save error: {str(db_error)}")
            # Continue processing even if DB save fails
        
        return result.get('structured_data', {}), result.get('metadata', {})
    
    except Exception as e:
        logging.getLogger(__name__).error(f"Unhandled error in process_pdf: {str(e)}", exc_info=True)
        return {"error": str(e)}, {"error": str(e)}
        
    # Close the file wrapper if needed
    if isinstance(uploaded_file, str) and file_wrapper:
        try:
            file_wrapper.close()
        except:
            pass

def process_pdf_batch(batch):
    """
    Process a batch of PDF files in parallel
    
    Args:
        batch: List of Streamlit uploaded file objects or file paths
        
    Returns:
        list: List of results for each PDF
    """
    if not batch:
        return []
    
    batch_results = []
    
    try:
        # Process with ThreadPoolExecutor for performance
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            # Submit all PDF processing tasks
            future_to_file = {executor.submit(process_pdf, file): file for file in batch}
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_file):
                file = future_to_file[future]
                
                try:
                    # Get results
                    result, metadata = future.result()
                    
                    # Add file info
                    if isinstance(file, str):
                        filename = os.path.basename(file)
                    else:
                        filename = file.name
                        
                    metadata['filename'] = filename
                    
                    # Combine results
                    combined_result = {**result, **metadata}
                    batch_results.append(combined_result)
                    
                    # Log success
                    logging.getLogger(__name__).info(f"Successfully processed {filename} in batch")
                except Exception as e:
                    # Handle errors
                    logging.getLogger(__name__).error(f"Error processing {filename} in batch: {str(e)}")
                    batch_results.append({
                        'filename': filename,
                        'processing_status': 'error',
                        'error_message': str(e)
                    })
    
    except Exception as e:
        logging.getLogger(__name__).error(f"Batch processing error: {str(e)}")
    
    return batch_results

def process_page(page):
    """
    Process a single PDF page with OCR
    
    Args:
        page: PDF page object
        
    Returns:
        dict: OCR results and extracted text
    """
    try:
        # Get OCR engine
        ocr_engine = init_paddle()
        
        if ocr_engine:
            # Convert page to image
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            img_np = np.array(img)
            
            # Run OCR
            ocr_result = ocr_engine.ocr(img_np, cls=True)
            
            # Extract text
            text = ""
            for line in ocr_result:
                for word_info in line:
                    if isinstance(word_info, list) and len(word_info) >= 2:
                        text += word_info[1][0] + " "
            
            return {
                'ocr_result': ocr_result,
                'text': text
            }
        else:
            return {
                'error': 'OCR engine not available'
            }
    except Exception as e:
        logger.error(f"Error in process_page: {e}")
        return {
            'error': str(e)
        }

def debug_ocr_process(uploaded_file, debug_info=None):
    """
    Process PDF with detailed debugging information
    
    Args:
        uploaded_file: Streamlit uploaded file object or path to a PDF file
        debug_info: Optional dict to store debug information
        
    Returns:
        dict: Extraction results and debug information
    """
    if debug_info is None:
        debug_info = {}
    
    debug_info['stages'] = []
    
    try:
        # Create a file-like object for both string paths and uploaded files
        file_wrapper = None
        
        # Handle different input types
        if isinstance(uploaded_file, str):
            # Input is a file path
            if not os.path.exists(uploaded_file):
                raise FileNotFoundError(f"File not found: {uploaded_file}")
                
            # Create a file-like object
            class FileWrapper:
                def __init__(self, path):
                    self.path = path
                    self.name = os.path.basename(path)
                    self._file = open(path, 'rb')
                    self._content = None
                    
                def read(self):
                    self._file.seek(0)
                    if self._content is None:
                        self._content = self._file.read()
                    return self._content
                    
                def seek(self, pos):
                    self._file.seek(pos)
                    
                def getvalue(self):
                    if self._content is None:
                        self._content = self.read()
                    return self._content
                    
                def close(self):
                    self._file.close()
                    
                def __enter__(self):
                    return self
                    
                def __exit__(self, exc_type, exc_val, exc_tb):
                    self.close()
            
            file_wrapper = FileWrapper(uploaded_file)
        else:
            # Input is a Streamlit UploadedFile
            file_wrapper = uploaded_file
        
        # Extract text with multiple methods for comparison
        methods = []
        
        # Try pdfplumber
        if PDFPLUMBER_AVAILABLE:
            try:
                file_wrapper.seek(0)
                with pdfplumber.open(file_wrapper) as pdf:
                    pdfplumber_text = ""
                    for page in pdf.pages:
                        pdfplumber_text += page.extract_text() or ""
                    
                    methods.append({
                        'name': 'pdfplumber',
                        'text': pdfplumber_text,
                        'page_count': len(pdf.pages)
                    })
            except Exception as e:
                methods.append({
                    'name': 'pdfplumber',
                    'error': str(e)
                })
        
        # Try PyMuPDF (fitz)
        if PYMUPDF_AVAILABLE:
            try:
                file_wrapper.seek(0)
                with fitz.open(stream=file_wrapper.getvalue(), filetype="pdf") as pdf:
                    fitz_text = ""
                    for page in pdf:
                        fitz_text += page.get_text() or ""
                    
                    methods.append({
                        'name': 'pymupdf',
                        'text': fitz_text,
                        'page_count': len(pdf)
                    })
            except Exception as e:
                methods.append({
                    'name': 'pymupdf',
                    'error': str(e)
                })
        
        # Try pdf2image + OCR
        if PDF2IMAGE_AVAILABLE and PADDLE_AVAILABLE:
            try:
                file_wrapper.seek(0)
                images = convert_from_bytes(file_wrapper.getvalue())
                ocr_text = ""
                
                # Initialize PaddleOCR if needed
                ocr = init_paddle()
                
                for img in images:
                    # Convert PIL image to numpy array
                    img_array = np.array(img)
                    
                    # Run OCR
                    result = ocr.ocr(img_array, cls=True)
                    
                    # Extract text
                    page_text = ""
                    for line in result[0]:
                        if line[1][0]:  # Check if there is text
                            page_text += line[1][0] + " "
                    
                    ocr_text += page_text + "\n\n"
                
                methods.append({
                    'name': 'pdf2image+paddleocr',
                    'text': ocr_text,
                    'page_count': len(images)
                })
            except Exception as e:
                methods.append({
                    'name': 'pdf2image+paddleocr',
                    'error': str(e)
                })
                
        # Process with main function to get final results
        result, metadata = process_pdf(file_wrapper)
        
        # Close the file wrapper if needed
        if isinstance(uploaded_file, str) and file_wrapper:
            try:
                file_wrapper.close()
            except:
                pass
                
        return {
            'methods': methods,
            'final_result': result,
            'metadata': metadata
        }
        
    except Exception as e:
        return {
            'error': str(e),
            'methods': methods if 'methods' in locals() else []
        }

def extract_structured_data_from_text(text):
    """
    Extract structured data from OCR text.
    
    Args:
        text: OCR text extracted from a document
        
    Returns:
        Dictionary with structured data fields
    """
    if not text or not isinstance(text, str):
        logger.warning("Empty or invalid text provided for extraction")
        return {
            'patient_name': "N/A",
            'age': "N/A",
            'exam_date': "N/A",
            'clinical_history': "N/A",
            'patient_history': "N/A",
            'findings': "N/A",
            'impression': "N/A",
            'recommendation': "N/A",
            'mammograph_results': "N/A",
            'birads_score': "N/A",
            'facility': "N/A",
            'exam_type': "N/A",
            'referring_provider': "N/A",
            'interpreting_provider': "N/A",
            'raw_ocr_text': text or ""
        }
        
    try:
        # Use our new parser integration
        from document_processing.parser_integration import process_ocr_text
        
        # Extract structured data using the new parser
        structured_data = process_ocr_text(text)
        logger.info("Extracted structured data using the new parser")
        
        return structured_data
        
    except ImportError:
        # Fallback to the old extraction method if the new parser is not available
        logger.warning("New parser not available, falling back to legacy extraction")
        try:
            # Import locally to avoid circular imports
            from document_processing.text_analysis import process_document_text
            
            result = process_document_text(text)
            # Make sure raw_ocr_text is included
            result['raw_ocr_text'] = text
            return result
            
        except Exception as e:
            logger.error(f"Error in legacy text extraction: {str(e)}")
            return {
                'patient_name': "N/A",
                'exam_date': "N/A",
                'birads_score': "N/A",
                'raw_ocr_text': text
            }
    except Exception as e:
        logger.error(f"Error extracting structured data: {str(e)}")
        return {
            'patient_name': "N/A",
            'exam_date': "N/A",
            'birads_score': "N/A",
            'raw_ocr_text': text
        } 