"""
UI tabs for Medical Report Processor application.
Contains functions for rendering different application tabs.
"""
import os
import re
import sys
import json
import logging
import tempfile
import shutil
import base64
import numpy as np
import pandas as pd
import pdfplumber
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
from io import BytesIO
from PIL import Image
from datetime import datetime
from wordcloud import WordCloud
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple

# Add parent directory to path to fix import errors
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Fixed import: using absolute import
import ocr_utils

from document_processing.ocr import process_pdf, process_pdf_batch, debug_ocr_process
from document_processing.text_analysis import extract_sections
from models.inference import standardize_birads, extract_medical_terms, get_term_explanations
from database.operations import save_to_db, init_db, Session, Report
from ui.components import (
    render_message, display_report_card, display_pdf_image, 
    display_file_uploader, display_metrics, plot_data_overview
)

# Set up logging
logger = logging.getLogger(__name__)

def render_upload_tab(models):
    """
    Render the Upload tab for processing new documents
    
    Args:
        models: Dictionary of loaded models
    """
    st.header("OCR Processing")
    st.write("Upload your scanned mammogram reports (PDFs) to extract data.")
    
    # PDF Uploader
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type="pdf",
        accept_multiple_files=True
    )
    
    # Add debug option
    debug_mode = st.checkbox("Enable Debug Mode", value=True, 
                            help="Show raw OCR text and extraction details")
    
    # Add to OCR Processing tab
    psm_mode = st.selectbox(
        "Layout Analysis Mode",
        options=[
            ("Fully Automatic", "3"),
            ("Single Column", "4"), 
            ("Single Text Block", "6")
        ],
        format_func=lambda x: x[0],
        help="Adjust for different document layouts"
    )
    os.environ["TESS_PSM"] = psm_mode[1]
    
    if st.button("Process PDFs"):
        if not uploaded_files:
            st.warning("Please upload at least one PDF file")
        else:
            with st.spinner("Processing PDFs..."):
                # Initialize database
                init_db()
                
                # Create progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Create debug expander
                debug_expander = st.expander("Show Raw OCR Text (for debugging)")
                
                # Process each PDF as a complete document
                all_results = []
                total_files = len(uploaded_files)
                
                for i, file in enumerate(uploaded_files):
                    try:
                        # Update status
                        status_text.text(f"Processing {file.name} ({i+1}/{total_files})...")
                        progress_bar.progress((i) / total_files)
                        
                        # Create a temporary file to store the PDF content
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                            # Write the file data to the temporary file
                            file_bytes = file.getvalue()
                            temp_file.write(file_bytes)
                            temp_file_path = temp_file.name
                        
                        # Process the PDF using the temporary file path
                        try:
                            structured_data, metadata = process_pdf(temp_file_path)
                            
                            # Add filename to metadata
                            if 'filename' not in metadata:
                                metadata['filename'] = file.name
                            
                            # Combine structured data and metadata for the results
                            result = {}
                            result.update(structured_data)
                            result.update(metadata)
                            
                            # Show debug info if enabled
                            if debug_mode and 'raw_text' in metadata:
                                with debug_expander:
                                    st.text_area(
                                        f"Raw OCR Text - {file.name}", 
                                        metadata.get('raw_text', 'No text extracted'), 
                                        height=200,
                                        key=f"debug_text_{i}"
                                    )
                            
                            # Add to results
                            all_results.append(result)
                        except Exception as processing_error:
                            st.error(f"Error processing {file.name}: {str(processing_error)}")
                            # Add error entry
                            error_result = {
                                'filename': file.name,
                                'processing_status': 'error',
                                'error_message': str(processing_error)
                            }
                            all_results.append(error_result)
                        finally:
                            # Clean up the temporary file
                            try:
                                os.unlink(temp_file_path)
                            except Exception:
                                pass  # Ignore cleanup errors
                    except Exception as e:
                        st.error(f"Error handling {file.name}: {str(e)}")
                        # Add error entry
                        error_result = {
                            'filename': file.name,
                            'processing_status': 'error',
                            'error_message': str(e)
                        }
                        all_results.append(error_result)
                    
                    # Update progress
                    progress_bar.progress((i + 1) / total_files)
                
                # Complete progress
                progress_bar.progress(1.0)
                status_text.text("Processing complete!")
            
            # Create DataFrame with one row per PDF
            if all_results:
                # Ensure consistent columns
                required_columns = [
                    'birads_score', 'document_date', 'document_type', 
                    'electronically_signed_by', 'exam_date', 'impression_result', 
                    'mammogram_results', 'patient_history', 'patient_name', 
                    'recommendation', 'testing_provider', 'ultrasound_results',
                    'raw_ocr_text'  # Add raw OCR text column
                ]
                
                # Create DataFrame with consistent columns
                df = pd.DataFrame(all_results)
                
                # Add missing columns with default values
                for col in required_columns:
                    if col not in df.columns:
                        df[col] = "Not Available"
                
                # Replace empty values with "Not Available"
                df = df.fillna("Not Available")
                
                # Handle redacted patient names - if patient_name is "Not Available" and raw_text contains 
                # patterns indicating redaction like "Patient: " followed by no name or "Patient Name:"
                # followed by no identifiable name
                def detect_redacted_patient(row):
                    if row['patient_name'] == "Not Available" or row['patient_name'] == "":
                        raw_text = row.get('raw_ocr_text', '')
                        redaction_patterns = [
                            r'Patient:\s*$',
                            r'Patient:\s*\n',
                            r'Patient Name:\s*$',
                            r'Patient Name:\s*\n',
                            r'Patient:.*redacted',
                            r'Patient:.*PROTECTED',
                            r'Name:.*redacted',
                            r'NAME:.*redacted'
                        ]
                        for pattern in redaction_patterns:
                            if re.search(pattern, raw_text, re.IGNORECASE):
                                return "Redacted"
                        
                            # Check for obvious medical document but no patient name found
                            medical_indicators = [
                                'MAMMOGRAM', 'ULTRASOUND', 'BIRADS', 'REPORT', 'SCREENING', 
                                'MEDICAL IMAGING', 'DIAGNOSTIC', 'RADIOLOGY'
                            ]
                            for indicator in medical_indicators:
                                if indicator in raw_text.upper() and len(row['patient_name']) < 3:
                                    return "Redacted"
                    return row['patient_name']
                
                # Apply redaction detection
                df['patient_name'] = df.apply(detect_redacted_patient, axis=1)
                
                # Process JSON-formatted fields and extract values with better error handling
                json_fields = ['provider_info', 'facility', 'patient_history', 'birads_score']
                for col in json_fields:
                    if col in df.columns:
                        df[col] = df.apply(
                            lambda row: _extract_json_value(row[col], col),
                            axis=1
                        )
                
                # Handle provider information specifically
                if 'provider_info' in df.columns:
                    # Extract provider name from provider_info if it exists
                    df['provider_name'] = df['provider_info'].apply(
                        lambda x: _extract_provider_name(x)
                    )
                    
                    # Extract referring provider from provider_info if it exists
                    df['referring_provider'] = df['provider_info'].apply(
                        lambda x: _extract_referring_provider(x)
                    )
                    
                    # If testing_provider is empty but we have provider_name, use that
                    df['testing_provider'] = df.apply(
                        lambda row: row['provider_name'] 
                            if row['testing_provider'] == "Not Available" and row['provider_name'] 
                            else row['testing_provider'],
                        axis=1
                    )
                
                # Enhanced extraction of key medical information based on raw text
                df = _enhance_field_extraction(df)
                
                # Improved deduplication with content merging rather than just references
                duplicate_pairs = [
                    ('impression_result', 'clinical_history'),
                    ('impression_result', 'findings'),
                    ('patient_history', 'clinical_history'),
                    ('electronically_signed_by', 'testing_provider')
                ]
                
                for field1, field2 in duplicate_pairs:
                    if field1 in df.columns and field2 in df.columns:
                        # If first field is empty but second has content, copy content
                        df[field1] = df.apply(
                            lambda row: row[field2] 
                                if row[field1] == "Not Available" and row[field2] != "Not Available" 
                                else row[field1],
                            axis=1
                        )
                        
                        # If both have content and they're different, merge them
                        # (instead of just referencing one from the other)
                        df[field1] = df.apply(
                            lambda row: _merge_field_content(row[field1], row[field2])
                                if row[field1] != "Not Available" and row[field2] != "Not Available" and row[field1] != row[field2]
                                else row[field1],
                            axis=1
                        )
                        
                        # Mark the second field as reference to first only if they're duplicates
                        df[field2] = df.apply(
                            lambda row: "See " + field1.replace('_', ' ').title() 
                                if _is_duplicate_content(row[field1], row[field2]) and row[field1] != "Not Available"
                                else row[field2],
                            axis=1
                        )
                
                # Store in session state
                st.session_state.df = df
                
                # Display results with tabs
                subtab1, subtab2, subtab3, subtab4 = st.tabs(["Full Data", "Structured Summary", "Raw OCR Text", "Debug Info"])
                
                with subtab1:
                    st.subheader("Full Extracted Medical Data")
                    
                    # Organize columns in a logical manner
                    preferred_column_order = [
                        'patient_name', 'document_type', 'exam_date', 'document_date',
                        'birads_score', 'impression_result', 'recommendation',
                        'mammogram_results', 'ultrasound_results', 'patient_history',
                        'electronically_signed_by', 'testing_provider', 
                        'md5_hash', 'filename'
                    ]
                    
                    # Filter to only include columns that exist in our dataframe
                    display_columns = [col for col in preferred_column_order if col in df.columns]
                    
                    # Add any remaining columns that weren't in our preferred order
                    for col in df.columns:
                        if col not in display_columns and col != 'raw_ocr_text':
                            display_columns.append(col)
                    
                    # Create a display dataframe with readable column names
                    display_df = df[display_columns].copy()
                    
                    # Rename columns for better readability
                    column_rename = {
                        'patient_name': 'Patient Name',
                        'document_type': 'Exam Type',
                        'exam_date': 'Exam Date',
                        'document_date': 'Report Date',
                        'birads_score': 'BIRADS Score',
                        'impression_result': 'Impression',
                        'recommendation': 'Recommendations',
                        'mammogram_results': 'Mammogram Results',
                        'ultrasound_results': 'Ultrasound Results',
                        'patient_history': 'Patient History',
                        'electronically_signed_by': 'Signed By',
                        'testing_provider': 'Provider',
                        'md5_hash': 'MD5 Hash',
                        'filename': 'Filename'
                    }
                    
                    # Apply renames that exist in our columns
                    rename_dict = {k: v for k, v in column_rename.items() if k in display_df.columns}
                    display_df = display_df.rename(columns=rename_dict)
                    
                    # Display the organized dataframe
                    st.dataframe(display_df, use_container_width=True)
                    
                    # Add export and save options
                    col1, col2 = st.columns(2)
                    with col1:
                        # Add export to CSV button
                        csv = df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "📥 Export to CSV",
                            data=csv,
                            file_name="medical_reports.csv",
                            mime="text/csv",
                            help="Download all reports as CSV file"
                        )
                    
                    with col2:
                        # Add save to database button
                        if st.button("💾 Save All to Database"):
                            with st.spinner("Saving to database..."):
                                saved_count = 0
                                for i, row in df.iterrows():
                                    try:
                                        # Prepare the data for database
                                        save_to_db({
                                            "md5_hash": row.get('md5_hash', ''),
                                            "filename": row.get('filename', 'Unknown'),
                                            "patient_name": row.get('patient_name', 'Not Available'),
                                            "exam_date": row.get('exam_date', 'Not Available'),
                                            "findings": row.to_dict(),
                                            "metadata": {
                                                "md5_hash": row.get('md5_hash', ''),
                                                "filename": row.get('filename', 'Unknown'),
                                                "exam_type": row.get('document_type', 'Unknown'),
                                                "processing_date": row.get('document_date', '')
                                            },
                                            "raw_ocr_text": row.get('raw_ocr_text', '')
                                        })
                                        saved_count += 1
                                    except Exception as e:
                                        logger.error(f"Failed to save report {row.get('filename', 'Unknown')}: {e}")
                                        st.error(f"Error saving {row.get('filename', 'Unknown')}: {str(e)}")
                                    
                                    st.success(f"Successfully saved {saved_count} of {len(df)} reports to database")
                
                with subtab2:
                    st.subheader("Structured Summary")
                    
                    # Create a compact display table with specific columns in preferred order
                    summary_columns = [
                        'patient_name', 'birads_score', 'document_type', 
                        'exam_date', 'document_date', 'testing_provider',
                        'electronically_signed_by'
                    ]
                    
                    # Ensure all required columns exist by filtering
                    summary_columns = [col for col in summary_columns if col in df.columns]
                    
                    # Create a clean display dataframe with better column names
                    summary_df = df[summary_columns].copy()
                    
                    # Rename columns for better display
                    summary_rename = {
                        'patient_name': 'Patient Name',
                        'birads_score': 'BIRADS',
                        'document_type': 'Exam Type',
                        'exam_date': 'Exam Date',
                        'document_date': 'Report Date',
                        'testing_provider': 'Provider',
                        'electronically_signed_by': 'Signed By'
                    }
                    
                    # Apply renames that exist in our columns
                    summary_rename_dict = {k: v for k, v in summary_rename.items() if k in summary_df.columns}
                    summary_df = summary_df.rename(columns=summary_rename_dict)
                    
                    # Display the nicely formatted dataframe
                    st.dataframe(summary_df, use_container_width=True)
                
                with subtab3:
                    # Show raw OCR text for selected document
                    st.subheader("View Raw OCR Text")
                    st.write("Select a document to view the extracted text:")
                    
                    # Create a more user-friendly selector
                    if 'patient_name' in df.columns and 'document_date' in df.columns:
                        # Create nice document labels
                        doc_labels = []
                        for idx, row in df.iterrows():
                            patient = row.get('patient_name', 'Unknown Patient')
                            date = row.get('document_date', 'Unknown Date')
                            doc_type = row.get('document_type', 'Unknown Type')
                            doc_labels.append(f"{patient} - {doc_type} ({date})")
                    else:
                        # Fallback to filenames if structured data not available
                        doc_labels = [f"Document {i+1}: {row.get('filename', f'File {i+1}')}" 
                                     for i, row in df.iterrows()]
                    
                    # Display selector with improved styling
                    selected_doc_idx = st.selectbox(
                        "Select document:",
                        options=range(len(doc_labels)),
                        format_func=lambda i: doc_labels[i]
                    )
                    
                    if selected_doc_idx is not None:
                        # Get raw text from the selected document
                        raw_text = df.iloc[selected_doc_idx].get('raw_ocr_text', 'No text available')
                        
                        # Display in a nicer format
                        st.markdown("### Raw Extracted Text")
                        
                        # Add copy button for convenience
                        st.download_button(
                            "Copy Text to Clipboard",
                            data=raw_text,
                            file_name="ocr_text.txt",
                            mime="text/plain"
                        )
                        
                        # Show text in a scrollable area with monospace font
                        st.text_area(
                            "",  # No label to avoid duplication
                            value=raw_text,
                            height=400,
                            key="selected_doc_ocr_text"
                        )
                
                with subtab4:
                    st.subheader("Extraction Debug Information")
                    st.write("Field extraction success rate:")
                    
                    # Calculate extraction success
                    extraction_stats = {}
                    for col in required_columns:
                        if col != 'raw_ocr_text':
                            success_rate = (df[col] != "Not Available").mean() * 100
                            extraction_stats[col] = success_rate
                    
                    # Display as bar chart
                    stats_df = pd.DataFrame(list(extraction_stats.items()), columns=['Field', 'Success Rate (%)'])
                    st.bar_chart(stats_df.set_index('Field'))
            else:
                st.warning("No valid data extracted from PDFs")
        
        # If debug mode is enabled, show raw text immediately
        if debug_mode and uploaded_files:
            st.subheader("Raw OCR Text (Debug)")
            for i, file in enumerate(uploaded_files):
                with pdfplumber.open(file) as pdf:
                    for j, page in enumerate(pdf.pages):
                        page_text = page.extract_text() or "No text extracted with pdfplumber"
                        st.text_area(
                            f"{file.name} - Page {j+1}",
                            page_text, 
                            height=200,
                            key=f"debug_text_{i}_{j}"  # Unique key combining file and page index
                        )
    
    # Display results
    if 'df' in st.session_state:
        st.subheader("Processed Results")
        
        # Display structured data only if we have results
        st.subheader("Structured Medical Data")
        if not st.session_state['df'].empty:
            try:
                # Allow user to select which result to view in detail
                if len(st.session_state['df']) > 1:
                    selected_index = st.selectbox(
                        "Select file to view details:",
                        range(len(st.session_state['df'])),
                        format_func=lambda i: st.session_state['df'].iloc[i].get('filename', f"Result {i+1}")
                    )
                else:
                    selected_index = 0
                
                # Convert the selected result to structured JSON format
                selected_row = st.session_state['df'].iloc[selected_index].to_dict()
                
                # Make sure we have ocr_utils module available
                if 'ocr_utils' in sys.modules:
                    structured_data = ocr_utils.convert_to_structured_json(selected_row)
                else:
                    # Fallback if ocr_utils not available
                    structured_data = {
                        "patient": {
                            "name": selected_row.get("patient_name", "Not Available"),
                            "id": selected_row.get("patient_id", "Not Available"),
                            "date_of_birth": selected_row.get("date_of_birth", "Not Available")
                        },
                        "exam": {
                            "type": selected_row.get("exam_type", "Not Available"),
                            "date": selected_row.get("exam_date", "Not Available"),
                            "birads_score": selected_row.get("birads_score", "Not Available")
                        },
                        "findings": selected_row.get("findings", "Not Available"),
                        "impression": selected_row.get("impression_result", "Not Available"),
                        "recommendation": selected_row.get("recommendation", "Not Available"),
                        "provider": {
                            "name": selected_row.get("testing_provider", "Not Available"),
                            "referring": selected_row.get("referring_provider", "Not Available")
                        }
                    }
                
                # Display the structured data
                st.json(structured_data)
            except Exception as e:
                st.error(f"Error generating structured data: {str(e)}")
                # Show raw data as fallback
                try:
                    st.json(st.session_state['df'].iloc[0].to_dict())
                except:
                    st.warning("Could not display row data.")
        else:
            st.warning("No structured data available - DataFrame is empty")
        
        # Display full dataframe with all fields
        st.subheader("Full Extracted Data")
        st.dataframe(st.session_state['df'], use_container_width=True)
        
        # Enhanced CSV download
        csv = st.session_state['df'].to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download Full Report",
            data=csv,
            file_name="full_mammogram_report.csv",
            mime="text/csv",
            help="Includes all extracted fields and confidence scores"
        )

    # Check Tesseract installation
    try:
        if not ocr_utils.check_tesseract_installation():
            st.error("""
            ### Tesseract OCR is not installed or not in your PATH
            
            #### Windows Installation:
            1. Download the installer from [UB-Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)
            2. Install with default options
            3. Add to PATH: `C:\\Program Files\\Tesseract-OCR`
            4. Restart the application
            
            #### Mac Installation:
            ```
            brew install tesseract
            ```
            
            #### Linux Installation:
            ```
            sudo apt-get install tesseract-ocr
            ```
            """)
    except Exception as e:
        st.warning(f"Could not check Tesseract installation: {str(e)}")

    # Debug OCR expander
    with st.expander("Debug OCR"):
        if st.button("Run OCR Diagnostics"):
            if uploaded_files:
                try:
                    debug_result = debug_ocr_process(uploaded_files[0])
                    st.json(debug_result)
                except Exception as e:
                    st.error(f"Diagnostics error: {str(e)}")
            else:
                st.warning("Please upload a PDF first")

def _extract_json_value(value, field_name):
    """
    Extract a value from a JSON string or return the original value if not JSON.
    
    Args:
        value: The value to extract from (could be JSON string or regular value)
        field_name: The field name to extract from JSON
        
    Returns:
        Extracted value or original value
    """
    if not value or not isinstance(value, str):
        return value
        
    # Check if value is a JSON string
    if value.strip().startswith('{') and value.strip().endswith('}'):
        try:
            data = json.loads(value)
            # If field_name is in the JSON, return that value
            if field_name in data:
                return data[field_name]
            # If 'value' is in the JSON, return that
            elif 'value' in data:
                return data['value']
            # Return the first value found
            elif data:
                return next(iter(data.values()))
        except json.JSONDecodeError:
            pass
    
    return value

def _extract_provider_name(provider_info):
    """
    Extract provider name from provider information.
    
    Args:
        provider_info: Provider information (could be JSON string, dict, or string)
        
    Returns:
        Provider name or empty string if not found
    """
    if not provider_info:
        return ""
        
    # If provider_info is a string that might be JSON
    if isinstance(provider_info, str):
        # Try to parse as JSON
        if provider_info.strip().startswith('{'):
            try:
                data = json.loads(provider_info)
                # Check common provider fields
                for field in ['name', 'provider_name', 'value', 'electronically_signed_by']:
                    if field in data and data[field]:
                        return data[field]
            except json.JSONDecodeError:
                pass
                
        # If not JSON or JSON parsing failed, check for common patterns
        provider_patterns = [
            r'(?:Dr\.?|MD|Doctor)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
            r'(?:electronically\s+signed\s+by|signed\s+by)[:\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
            r'(?:radiologist|physician|provider)[:\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)'
        ]
        
        for pattern in provider_patterns:
            match = re.search(pattern, provider_info, re.IGNORECASE)
            if match:
                return match.group(1).strip()
    
    # If provider_info is a dict
    elif isinstance(provider_info, dict):
        # Check common provider fields
        for field in ['name', 'provider_name', 'value', 'electronically_signed_by']:
            if field in provider_info and provider_info[field]:
                return provider_info[field]
    
    # Return original if no extraction was possible
    return provider_info if isinstance(provider_info, str) else ""

def _extract_referring_provider(provider_info):
    """
    Extract referring provider from provider information.
    
    Args:
        provider_info: Provider information (could be JSON string, dict, or string)
        
    Returns:
        Referring provider name or empty string if not found
    """
    if not provider_info:
        return ""
        
    # If provider_info is a string that might be JSON
    if isinstance(provider_info, str):
        # Try to parse as JSON
        if provider_info.strip().startswith('{'):
            try:
                data = json.loads(provider_info)
                # Check common referring provider fields
                for field in ['referring_provider', 'referring', 'referrer']:
                    if field in data and data[field]:
                        return data[field]
            except json.JSONDecodeError:
                pass
                
        # If not JSON or JSON parsing failed, check for common patterns
        referring_patterns = [
            r'(?:referring\s+(?:physician|provider|doctor))[:\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
            r'(?:referred\s+by)[:\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)'
        ]
        
        for pattern in referring_patterns:
            match = re.search(pattern, provider_info, re.IGNORECASE)
            if match:
                return match.group(1).strip()
    
    # If provider_info is a dict
    elif isinstance(provider_info, dict):
        # Check common referring provider fields
        for field in ['referring_provider', 'referring', 'referrer']:
            if field in provider_info and provider_info[field]:
                return provider_info[field]
    
    # Return empty string if no extraction was possible
    return ""

def _enhance_field_extraction(df):
    """
    Enhance field extraction based on raw text and other fields.
    
    Args:
        df: DataFrame with extracted fields
        
    Returns:
        Enhanced DataFrame
    """
    # Make a copy to avoid modifying the original
    enhanced_df = df.copy()
    
    # Process each row
    for idx, row in enhanced_df.iterrows():
        # Get raw OCR text if available
        raw_text = row.get('raw_ocr_text', '')
        
        # Skip if no raw text
        if not raw_text:
            continue
            
        # Import necessary functions
        try:
            from document_processing.text_analysis import extract_section, extract_birads_score
            from ocr_utils import normalize_date
        except ImportError:
            # If imports fail, use regex fallbacks
            continue
            
        # Enhance BIRADS score - extract just the number
        if not row.get('birads_score') or row.get('birads_score') == 'Not Available':
            birads_result = extract_birads_score(raw_text)
            if birads_result and birads_result.get('value'):
                enhanced_df.at[idx, 'birads_score'] = birads_result['value']
                
        # Enhance dates with normalization
        for date_field in ['exam_date', 'document_date']:
            if row.get(date_field):
                enhanced_df.at[idx, date_field] = normalize_date(row[date_field])
                
        # Enhance sections with better extraction
        for section_field, section_name in [
            ('findings', 'findings'),
            ('impression_result', 'impression'),
            ('recommendation', 'recommendation'),
            ('patient_history', 'clinical_history')
        ]:
            if not row.get(section_field) or row.get(section_field) == 'Not Available':
                section_text = extract_section(raw_text, section_name)
                if section_text:
                    enhanced_df.at[idx, section_field] = section_text
                    
        # Merge clinical_history and patient_history if both exist
        clinical_history = row.get('clinical_history', '')
        patient_history = row.get('patient_history', '')
        
        if clinical_history and patient_history and clinical_history != patient_history:
            merged_history = _merge_field_content(clinical_history, patient_history)
            enhanced_df.at[idx, 'patient_history'] = merged_history
            
        # Extract provider information
        provider_info = row.get('provider_info', '')
        if provider_info:
            provider_name = _extract_provider_name(provider_info)
            if provider_name and (not row.get('electronically_signed_by') or row.get('electronically_signed_by') == 'Not Available'):
                enhanced_df.at[idx, 'electronically_signed_by'] = provider_name
                
            referring_provider = _extract_referring_provider(provider_info)
            if referring_provider and (not row.get('referring_provider') or row.get('referring_provider') == 'Not Available'):
                enhanced_df.at[idx, 'referring_provider'] = referring_provider
    
    return enhanced_df

def _extract_section(text, start_markers, end_markers=None):
    """
    Extract a section from text using start and end markers.
    
    Args:
        text: Text to extract section from
        start_markers: List of regex patterns for section start
        end_markers: Optional list of regex patterns for section end
        
    Returns:
        Extracted section text or empty string if not found
    """
    if not text or not start_markers:
        return ""
        
    # Try each start marker
    for start_marker in start_markers:
        start_match = re.search(start_marker, text, re.IGNORECASE)
        if not start_match:
            continue
            
        # Get start position (end of the marker)
        start_pos = start_match.end()
        
        # If end markers provided, find the first one after start_pos
        if end_markers:
            # Combine end markers into a single pattern
            end_pattern = '|'.join(f'({marker})' for marker in end_markers)
            end_match = re.search(end_pattern, text[start_pos:], re.IGNORECASE)
            
            if end_match:
                end_pos = start_pos + end_match.start()
                return text[start_pos:end_pos].strip()
        
        # If no end markers or none found, look for the next section header
        next_section_pattern = r'\n(?:[A-Z][A-Z\s]{3,}:|\n\s*[A-Z][A-Z\s]{3,}:)'
        next_match = re.search(next_section_pattern, text[start_pos:])
        
        if next_match:
            end_pos = start_pos + next_match.start()
            return text[start_pos:end_pos].strip()
        else:
            # If no next section, return the rest of the text
            return text[start_pos:].strip()
    
    # Section not found
    return ""

def _merge_field_content(content1, content2):
    """
    Merge two field contents, avoiding duplication.
    
    Args:
        content1: First content
        content2: Second content
        
    Returns:
        Merged content
    """
    if not content1:
        return content2
    if not content2:
        return content1
        
    # Check if one is a subset of the other
    if content1 in content2:
        return content2
    if content2 in content1:
        return content1
        
    # Check if they're duplicates with minor differences
    if _is_duplicate_content(content1, content2):
        return content1 if len(content1) >= len(content2) else content2
        
    # Otherwise, combine them
    return f"{content1}\n{content2}"

def _is_duplicate_content(content1, content2):
    """
    Check if two contents are duplicates with minor differences.
    
    Args:
        content1: First content
        content2: Second content
        
    Returns:
        True if contents are duplicates, False otherwise
    """
    if not content1 or not content2:
        return False
        
    # Normalize for comparison
    norm1 = re.sub(r'\s+', ' ', content1.lower()).strip()
    norm2 = re.sub(r'\s+', ' ', content2.lower()).strip()
    
    # Check for exact match after normalization
    if norm1 == norm2:
        return True
        
    # Check if one is a subset of the other
    if norm1 in norm2 or norm2 in norm1:
        return True
        
    # Check similarity ratio
    from difflib import SequenceMatcher
    similarity = SequenceMatcher(None, norm1, norm2).ratio()
    return similarity > 0.8

def render_analysis_tab(models):
    """
    Render the Analysis tab for analyzing processed documents
    
    Args:
        models: Dictionary of loaded models
    """
    st.header("Analyze Processed Documents")
    
    # Check if we have data to analyze
    if 'df' in st.session_state and not st.session_state.df.empty:
        # Get the dataframe with processed documents
        df = st.session_state.df
        
        st.subheader(f"Batch Analysis ({len(df)} documents)")
        
        # Display metrics
        st.subheader("Summary Metrics")
        
        # Convert BIRADS score to numeric for averaging, handling errors gracefully
        try:
            birads_scores = pd.to_numeric(df["birads_score"], errors='coerce')
            avg_birads = round(birads_scores.mean(), 1)
        except:
            avg_birads = "N/A"
        
        # Basic metrics
        metrics = {
            "Total Documents": len(df),
            "Unique Exam Types": df["document_type"].nunique() if "document_type" in df.columns else 0,
            "Average BIRADS": avg_birads
        }
        
        # Display metrics in a nice format
        cols = st.columns(3)
        for i, (key, value) in enumerate(metrics.items()):
            cols[i].metric(key, value)
        
        # Create tabs for different analyses
        analysis_tab1, analysis_tab2, analysis_tab3 = st.tabs(["Data Overview", "Document Selection", "Terminology Analysis"])
        
        with analysis_tab1:
            # Display data overview and visualizations
            st.subheader("Data Overview")
            
            # BIRADS distribution
            if "birads_score" in df.columns:
                try:
                    birads_counts = df["birads_score"].value_counts().reset_index()
                    birads_counts.columns = ["BIRADS Score", "Count"]
                    
                    # Display as bar chart
                    st.subheader("BIRADS Score Distribution")
                    fig = px.bar(birads_counts, x="BIRADS Score", y="Count", 
                                text="Count", title="BIRADS Score Distribution")
                    fig.update_layout(xaxis_title="BIRADS Score", yaxis_title="Number of Documents")
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Could not generate BIRADS visualization: {str(e)}")
            
            # Exam type distribution
            if "document_type" in df.columns:
                try:
                    exam_counts = df["document_type"].value_counts().reset_index()
                    exam_counts.columns = ["Exam Type", "Count"]
                    
                    st.subheader("Exam Type Distribution")
                    fig = px.pie(exam_counts, values="Count", names="Exam Type", 
                                title="Exam Type Distribution")
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Could not generate exam type visualization: {str(e)}")
            
            # Findings wordcloud if we have text data
            if "raw_ocr_text" in df.columns:
                try:
                    st.subheader("Medical Term Word Cloud")
                    
                    # Instead of using all text, extract only medical terms for the word cloud
                    try:
                        # Get all document text
                        all_text = " ".join(df["raw_ocr_text"].astype(str).tolist())
                        
                        # Extract medical terms
                        medical_terms = extract_medical_terms(pd.Series([all_text]))
                        
                        if medical_terms and any(terms for terms in medical_terms.values()):
                            # Combine all medical terms with their frequency as weight
                            medical_term_text = ""
                            for category, terms in medical_terms.items():
                                for term in terms:
                                    # Add each term multiple times based on frequency in text
                                    count = len(re.findall(r'\b' + re.escape(term) + r'\b', all_text, re.IGNORECASE))
                                    medical_term_text += (term + " ") * (count if count > 0 else 1)
                            
                            # Generate and display word cloud with medical terms only
                            if medical_term_text:
                                wordcloud = WordCloud(width=800, height=400, background_color='white',
                                                     max_words=100, contour_width=3, 
                                                     collocations=False).generate(medical_term_text)
                                
                                fig, ax = plt.subplots(figsize=(10, 5))
                                ax.imshow(wordcloud, interpolation='bilinear')
                                ax.axis("off")
                                plt.title("Medical Terminology Frequency")
                                st.pyplot(fig)
                            else:
                                st.info("No medical terms identified for word cloud generation")
                        else:
                            st.info("No medical terms extracted from the documents")
                    except Exception as e:
                        st.warning(f"Could not generate medical term word cloud: {str(e)}")
                        
                        # Fallback to standard word cloud but with common words filtered
                        st.subheader("Document Text Word Cloud (Filtered)")
                        
                        # Get all text
                        all_text = " ".join(df["raw_ocr_text"].astype(str).tolist())
                        
                        # Filter out common non-medical words
                        common_words = ['the', 'and', 'to', 'of', 'a', 'in', 'for', 'is', 'on', 'that', 'by', 'this', 
                                       'with', 'i', 'you', 'it', 'not', 'or', 'be', 'are', 'from', 'at', 'as', 'your',
                                       'have', 'was', 'date', 'time', 'name', 'patient', 'report', 'doctor', 'medical']
                        
                        # Process text - lowercase and remove common words
                        processed_text = ' '.join([word for word in all_text.lower().split() 
                                                 if word not in common_words and len(word) > 3])
                        
                        # Generate and display filtered word cloud
                        wordcloud = WordCloud(width=800, height=400, background_color='white',
                                             max_words=100, contour_width=3).generate(processed_text)
                        
                        fig, ax = plt.subplots(figsize=(10, 5))
                        ax.imshow(wordcloud, interpolation='bilinear')
                        ax.axis("off")
                        st.pyplot(fig)
                except Exception as e:
                    st.error(f"Could not generate word cloud: {str(e)}")
        
        with analysis_tab2:
            # Document selector for individual analysis
            st.subheader("Individual Document Analysis")
            
            # Create document selector
            doc_labels = []
            for idx, row in df.iterrows():
                patient = row.get('patient_name', 'Unknown Patient')
                date = row.get('document_date', row.get('exam_date', 'Unknown Date'))
                doc_type = row.get('document_type', 'Unknown Type')
                doc_labels.append(f"{patient} - {doc_type} ({date})")
            
            selected_doc_idx = st.selectbox(
                "Select document to analyze:",
                options=range(len(doc_labels)),
                format_func=lambda i: doc_labels[i]
            )
            
            if selected_doc_idx is not None:
                # Get selected document data
                selected_doc = df.iloc[selected_doc_idx].to_dict()
                
                # Display key information
                st.subheader("Document Information")
                
                # Create a metrics display for key fields
                metric_cols = st.columns(3)
                
                metric_cols[0].metric("Patient Name", selected_doc.get('patient_name', 'Not Available'))
                metric_cols[1].metric("Exam Type", selected_doc.get('document_type', 'Not Available'))
                metric_cols[2].metric("BIRADS Score", selected_doc.get('birads_score', 'Not Available'))
                
                # Display document sections
                st.subheader("Document Content")
                
                sections_to_display = {
                    "Impression": selected_doc.get('impression_result', 'Not Available'),
                    "Mammogram Results": selected_doc.get('mammogram_results', 'Not Available'),
                    "Ultrasound Results": selected_doc.get('ultrasound_results', 'Not Available'),
                    "Recommendation": selected_doc.get('recommendation', 'Not Available'),
                    "Patient History": selected_doc.get('patient_history', 'Not Available')
                }
                
                for section, content in sections_to_display.items():
                    if content and content != 'Not Available':
                        with st.expander(section, expanded=section == "Impression"):
                            st.markdown(content)
                
                # Display raw text if available
                if 'raw_ocr_text' in selected_doc and selected_doc['raw_ocr_text']:
                    with st.expander("Raw OCR Text", expanded=False):
                        st.text_area("", selected_doc['raw_ocr_text'], height=300)
        
        with analysis_tab3:
            st.subheader("Medical Terminology Analysis")
            
            # Extract terminology if model is available
            try:
                if 'raw_ocr_text' in df.columns:
                    # Get all document text
                    all_text = df['raw_ocr_text'].tolist()
                    
                    # Extract medical terms
                    medical_terms = extract_medical_terms(pd.Series(all_text))
                    
                    if medical_terms:
                        # Create tabs for different term categories
                        term_cats = list(medical_terms.keys())
                        if term_cats:
                            term_tabs = st.tabs([cat.title() for cat in term_cats])
                            
                            for i, (category, terms) in enumerate(medical_terms.items()):
                                with term_tabs[i]:
                                    if terms:
                                        # Count term frequencies
                                        term_counts = {}
                                        for term in terms:
                                            count = sum(str(text).lower().count(term.lower()) for text in all_text if text)
                                            term_counts[term] = count
                                        
                                        # Sort by frequency
                                        term_counts = dict(sorted(term_counts.items(), key=lambda x: x[1], reverse=True))
                                        
                                        # Display as bar chart for most common terms (top 15)
                                        top_terms = dict(list(term_counts.items())[:15])
                                        
                                        if top_terms:
                                            term_df = pd.DataFrame([{"Term": k, "Count": v} for k, v in top_terms.items()])
                                            
                                            fig = px.bar(term_df, x="Term", y="Count", text="Count",
                                                        title=f"Most Common {category.title()} Terms")
                                            fig.update_layout(xaxis_title="Term", yaxis_title="Frequency")
                                            st.plotly_chart(fig, use_container_width=True)
                                            
                                            # Show term explanations
                                            try:
                                                explanations = get_term_explanations(category)
                                                
                                                if explanations:
                                                    with st.expander("Term Explanations", expanded=False):
                                                        for term, count in top_terms.items():
                                                            if term in explanations:
                                                                st.markdown(f"**{term}** ({count} occurrences): {explanations[term]}")
                                                            else:
                                                                st.markdown(f"**{term}** ({count} occurrences)")
                                            except Exception as e:
                                                st.warning(f"Could not load term explanations: {str(e)}")
                                    else:
                                        st.info(f"No {category} terms found")
                    else:
                        st.info("No medical terminology detected or terminology model not available")
                else:
                    st.warning("No text data available for terminology analysis")
            except Exception as e:
                st.error(f"Error analyzing medical terminology: {e}")
                logger.exception("Error in terminology analysis")
    else:
        # No data available
        st.info("No processed documents available for analysis. Please process documents in the OCR Processing tab first.")
        
        # Add a button to navigate to the upload tab
        if st.button("Go to OCR Processing"):
            # Set the active tab to Upload
            st.session_state.active_tab = "Upload"
            st.experimental_rerun()

def render_chatbot_tab(models):
    """
    Render the Chatbot tab for interacting with the AI assistant
    
    Args:
        models: Dictionary of loaded models
    """
    st.header("Medical Assistant Chatbot")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "I'm your medical document assistant. How can I help you today?"}
        ]
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about medical terminology or reports..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # First check if we have a document context
                    if 'single_result' in st.session_state:
                        # Add document context to prompt
                        result = st.session_state.single_result
                        
                        # Build a system prompt with document context
                        system_prompt = "You are a medical document assistant analyzing the following report:\n\n"
                        
                        if 'text' in result and result['text']:
                            system_prompt += f"REPORT TEXT:\n{result['text']}\n\n"
                        
                        if 'structured_data' in result and result['structured_data']:
                            system_prompt += "STRUCTURED DATA:\n"
                            for key, value in result['structured_data'].items():
                                system_prompt += f"{key}: {value}\n"
                        
                        system_prompt += "\nProvide accurate information based on this report. If asked about something not in the report, be clear about what information is available."
                        
                        # Get response from Claude with document context
                        if 'chatbot' in models:
                            response = models['chatbot'](prompt, st.session_state.messages, system_prompt)
                        else:
                            response = "Sorry, the AI assistant is not available at the moment."
                    else:
                        # General medical question without document context
                        if 'chatbot' in models:
                            response = models['chatbot'](prompt, st.session_state.messages)
                        else:
                            response = "Sorry, the AI assistant is not available at the moment."
                    
                    # Display the response
                    st.markdown(response)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_msg = f"Error generating response: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    logger.exception("Error in chatbot")

def render_database_tab():
    """
    Render the Database tab for viewing and managing saved reports
    """
    st.header("Database Management")
    
    # Get engine and session
    from database.operations import get_all_reports, get_report_by_hash, delete_report
    
    # Get all reports
    try:
        reports = get_all_reports()
        
        if reports:
            st.subheader(f"Saved Reports ({len(reports)})")
            
            # Create dataframe from reports
            report_data = []
            for report in reports:
                try:
                    # Parse JSON fields
                    findings = json.loads(report.findings) if report.findings else {}
                    metadata = json.loads(report.report_metadata) if report.report_metadata else {}
                    
                    report_data.append({
                        "ID": report.md5_hash,
                        "Filename": report.filename,
                        "Patient Name": report.patient_name,
                        "Exam Date": report.exam_date,
                        "Processed At": report.processed_at.strftime("%Y-%m-%d %H:%M:%S"),
                        "BIRADS Score": findings.get("birads_score", "N/A"),
                        "Exam Type": metadata.get("exam_type", "Unknown")
                    })
                except Exception as e:
                    logger.error(f"Error parsing report {report.md5_hash}: {e}")
                    # Add with minimal info if parsing fails
                    report_data.append({
                        "ID": report.md5_hash,
                        "Filename": report.filename,
                        "Patient Name": report.patient_name,
                        "Exam Date": report.exam_date,
                        "Processed At": report.processed_at.strftime("%Y-%m-%d %H:%M:%S"),
                        "BIRADS Score": "Error",
                        "Exam Type": "Error"
                    })
            
            # Create dataframe
            df = pd.DataFrame(report_data)
            
            # Display dataframe
            st.dataframe(df)
            
            # Create selectbox for report details
            selected_id = st.selectbox("Select a report to view details", options=df["ID"].tolist())
            
            if selected_id:
                # Get report details
                report = get_report_by_hash(selected_id)
                
                if report:
                    st.subheader(f"Report Details: {report.filename}")
                    
                    # Parse fields
                    findings = json.loads(report.findings) if report.findings else {}
                    metadata = json.loads(report.report_metadata) if report.report_metadata else {}
                    
                    # Display report information
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"**Patient Name:** {report.patient_name}")
                        st.markdown(f"**Exam Date:** {report.exam_date}")
                        st.markdown(f"**Processed At:** {report.processed_at.strftime('%Y-%m-%d %H:%M:%S')}")
                    
                    with col2:
                        st.markdown(f"**BIRADS Score:** {findings.get('birads_score', 'N/A')}")
                        st.markdown(f"**Exam Type:** {metadata.get('exam_type', 'Unknown')}")
                    
                    # Display raw OCR text
                    with st.expander("Raw OCR Text", expanded=False):
                        st.text_area("", report.raw_ocr_text, height=300)
                    
                    # Display findings
                    with st.expander("Findings", expanded=True):
                        st.json(findings)
                    
                    # Display metadata
                    with st.expander("Metadata", expanded=False):
                        st.json(metadata)
                    
                    # Delete report button
                    if st.button(f"Delete Report: {report.filename}"):
                        if delete_report(selected_id):
                            st.success(f"Report {report.filename} deleted successfully")
                            st.rerun()
                        else:
                            st.error("Failed to delete report")
                else:
                    st.error(f"Report with ID {selected_id} not found")
        else:
            st.info("No reports saved in the database")
    except Exception as e:
        st.error(f"Error loading reports: {e}")
        logger.exception("Error in database tab") 