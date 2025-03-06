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
import plotly.graph_objects as go
from io import BytesIO
from PIL import Image
from datetime import datetime
from wordcloud import WordCloud
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import inspect

# Add parent directory to path to fix import errors
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Fixed import: using absolute import
import ocr_utils

from document_processing.ocr import process_pdf, process_pdf_batch, debug_ocr_process
from document_processing.text_analysis import extract_sections
from models.inference import (
    standardize_birads, extract_medical_terms, get_term_explanations, 
    assess_report_quality, analyze_provider_quality
)
from database.operations import save_to_db, init_db, Session, Report
from ui.components import (
    render_message, display_report_card, display_pdf_image, 
    display_file_uploader, display_metrics, plot_data_overview
)

# Set up logging
logger = logging.getLogger(__name__)

def render_upload_tab(models):
    """
    Render the upload tab for processing medical documents.
    
    Args:
        models: Dictionary of loaded models
        
    Returns:
        None
    """
    st.header("Upload Medical Reports")
    
    # Settings
    with st.expander("Processing Settings", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            use_llm = st.checkbox("Use LLM Enhancement", value=False, 
                                 help="Use Large Language Model to enhance extraction (slower but more accurate)")
            debug_mode = st.checkbox("Debug Mode", value=True,  # Set default to True for now
                                    help="Show debugging information during processing")
                                    
        with col2:
            parallel_processing = st.checkbox("Parallel Processing", value=True,
                                            help="Process files in parallel for faster results")
            show_confidence = st.checkbox("Show Confidence Scores", value=False,
                                        help="Show confidence scores for extracted fields")
    
    # File upload
    uploaded_files = st.file_uploader("Upload Medical PDF Reports", 
                                     type=["pdf"], 
                                     accept_multiple_files=True,
                                     help="Upload one or more PDF medical reports for processing")
    
    # Add a debug tool option
    if debug_mode and uploaded_files:
        if st.button("Run OCR Diagnostics", type="secondary"):
            with st.spinner("Running diagnostics..."):
                # Import the OCR function and debug tools
                from document_processing.ocr import process_pdf
                from document_processing.debug_tools import test_extract_text
                
                # Run diagnostics on each file
                for file in uploaded_files:
                    st.subheader(f"Diagnostics for {file.name}")
                    
                    # Run the test
                    test_results = test_extract_text(process_pdf, file)
                    
                    # Display results
                    if test_results.get('success', False):
                        st.success(f"Successfully extracted text ({len(test_results.get('extracted_text', ''))} characters)")
                        st.write("Debug message:")
                        st.code(test_results.get('debug_message', ''))
                        st.write("Text preview:")
                        st.text_area("", test_results.get('extracted_text', '')[:1000] + "..." 
                                    if len(test_results.get('extracted_text', '')) > 1000 
                                    else test_results.get('extracted_text', ''),
                                    height=200)
                    else:
                        st.error("Failed to extract text")
                        if 'error' in test_results:
                            st.write("Error:", test_results['error'])
                            st.code(test_results.get('traceback', ''))
                    
                    # Show detailed debug info
                    with st.expander("Detailed OCR Result Structure"):
                        st.json(test_results.get('debug_info', {}))
    
    process_button = st.button("Process Reports", use_container_width=True, type="primary", 
                              disabled=len(uploaded_files) == 0)
    
    if process_button and uploaded_files:
        with st.spinner("Processing medical reports..."):
            # Process each uploaded file
            progress_bar = st.progress(0)
            results = []
            
            # Import original OCR processing function - KEEPING THE ORIGINAL OCR PROCESS
            from document_processing.ocr import process_pdf
            # Import only our new text analysis
            from document_processing.parser_integration import process_ocr_text
            # Import our debug tools
            from document_processing.debug_tools import extract_text_from_ocr_result
            
            # Process each file sequentially as in the original implementation
            for i, file in enumerate(uploaded_files):
                try:
                    # Update progress
                    progress_bar.progress((i) / len(uploaded_files))
                    
                    # Process with original OCR method - NO CHANGES TO OCR IMPLEMENTATION
                    st.write(f"Processing {file.name}...")
                    ocr_result = process_pdf(file)
                    
                    # Use our debug tool to extract text
                    extracted_text, debug_msg = extract_text_from_ocr_result(ocr_result)
                    
                    if debug_mode:
                        st.write("Extraction debug message:")
                        st.code(debug_msg)
                    
                    if extracted_text:
                        st.success(f"Successfully extracted {len(extracted_text)} characters from {file.name}")
                        
                        # Use our new parser for text analysis only
                        structured_data = process_ocr_text(extracted_text, use_llm=use_llm)
                        
                        # Add file information
                        structured_data['filename'] = file.name
                        
                        # Add to results
                        results.append(structured_data)
                        
                        # Debug information
                        if debug_mode:
                            st.text_area(
                                f"Raw OCR Text - {file.name}", 
                                extracted_text,
                                height=200,
                                key=f"debug_text_{i}"
                            )
                    else:
                        st.error(f"No text extracted from {file.name}")
                        if debug_mode:
                            st.write("Raw OCR result type:", type(ocr_result))
                            if isinstance(ocr_result, dict):
                                st.write("Keys:", list(ocr_result.keys()))
                except Exception as e:
                    st.error(f"Error processing {file.name}: {str(e)}")
                    if debug_mode:
                        import traceback
                        st.code(traceback.format_exc())
                
                # Update progress
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            # Create DataFrame from results
            if results:
                df = pd.DataFrame(results)
                
                # Convert any dictionary or non-standard types to strings for dataframe compatibility
                for col in df.columns:
                    df[col] = df[col].apply(
                        lambda x: str(x) if isinstance(x, (dict, list, tuple)) else x
                    )
                
                st.session_state['df'] = df
                
                # Show results summary
                st.success(f"Processed {len(results)} files successfully")
                
                # Show extraction statistics
                with st.expander("Extraction Statistics", expanded=True):
                    # Required columns for our application
                    required_columns = [
                        'patient_name', 'age', 'exam_date', 
                        'clinical_history', 'patient_history',
                        'findings', 'mammograph_results', 'impression',
                        'recommendation', 'birads_score', 
                        'facility', 'exam_type',
                        'referring_provider', 'interpreting_provider',
                        'raw_ocr_text'
                    ]
                    
                    # Calculate extraction success rate
                    extraction_stats = {}
                    
                    for col in required_columns:
                        if col in df.columns and col != 'raw_ocr_text':
                            # Count non-N/A values
                            success_rate = (df[col] != "N/A").mean() * 100
                            extraction_stats[col] = success_rate
                    
                    # Display extraction statistics
                    stats_df = pd.DataFrame(list(extraction_stats.items()), 
                                          columns=['Field', 'Success Rate (%)'])
                    st.bar_chart(stats_df.set_index('Field'))
                    
                    # Display the dataframe
                    report_cols = [col for col in required_columns if col != 'raw_ocr_text' and col in df.columns]
                    st.dataframe(df[report_cols])
                
                # Debug information
                if debug_mode:
                    st.subheader("Debug Information")
                    with st.expander("Raw Extracted Data", expanded=False):
                        for i, result in enumerate(results):
                            st.json(result)
            else:
                st.warning("No data was extracted from the uploaded files.")

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
        analysis_tab1, analysis_tab2, analysis_tab3, analysis_tab4 = st.tabs([
            "Data Overview", 
            "Document Selection", 
            "Terminology Analysis",
            "Report Quality Analysis"  # New tab
        ])
        
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
                    
                    # Detect language
                    language = 'en'  # Default to English
                    if 'language' in df.columns:
                        # Use the most common language in the dataset
                        language_counts = df['language'].value_counts()
                        if not language_counts.empty:
                            language = language_counts.index[0]
                    
                    # UI controls for filtering
                    st.sidebar.subheader("Terminology Analysis Options")
                    min_frequency = st.sidebar.slider("Minimum Term Frequency", 1, 10, 1, 
                                                    help="Only show terms that appear at least this many times")
                    
                    # Category selection
                    available_categories = [
                        "tissues", "characteristics", "signs", "diagnoses", 
                        "interventions", "measurements", "birads", "locations", 
                        "orientations", "procedures", "findings"
                    ]
                    selected_categories = st.sidebar.multiselect(
                        "Term Categories", 
                        available_categories,
                        default=available_categories[:5],  # Default to first 5 categories
                        help="Select which categories of medical terms to analyze"
                    )
                    
                    # Visualization options
                    viz_type = st.sidebar.radio(
                        "Visualization Type",
                        ["Bar Chart", "Word Cloud", "Table"],
                        index=0,
                        help="Select how to visualize the extracted terms"
                    )
                    
                    # Extract medical terms with language detection
                    medical_terms = extract_medical_terms(pd.Series(all_text), language=language)
                    
                    if not medical_terms:
                        st.info("No medical terminology detected or terminology model not available")
                        return
                        
                    # Filter by selected categories
                    if selected_categories:
                        medical_terms = {k: v for k, v in medical_terms.items() if k in selected_categories}
                    
                    if not medical_terms:
                        st.info("No medical terminology detected in the selected categories")
                        return
                        
                    # Show language information
                    st.info(f"Analyzing terminology in {'French' if language == 'fr' else 'English'} language")
                    
                    # No special handling for BIRADS anymore - we'll just treat it like other categories
                    # and display it with the visualization and term explanations
                    
                    # Get term categories
                    term_cats = list(medical_terms.keys())
                    
                    # Skip creating tabs if no categories
                    if not term_cats:
                        st.info("No terminology categories to display")
                        return
                    
                    # Create tabs for different term categories
                    term_tabs = st.tabs([cat.title() for cat in term_cats])
                    
                    for i, category in enumerate(term_cats):
                        terms = medical_terms[category]
                        with term_tabs[i]:
                            if not terms:
                                st.info(f"No {category} terms found")
                                continue
                                
                            # Count term frequencies
                            term_counts = {}
                            for term in terms:
                                count = sum(str(text).lower().count(term.lower()) for text in all_text if text)
                                if count >= min_frequency:  # Apply frequency filter
                                    term_counts[term] = count
                            
                            if not term_counts:
                                st.info(f"No {category} terms found with frequency ≥ {min_frequency}")
                                continue
                                
                            # Sort by frequency
                            term_counts = dict(sorted(term_counts.items(), key=lambda x: x[1], reverse=True))
                            
                            # Create dataframe for visualization
                            term_df = pd.DataFrame([{"Term": k, "Count": v} for k, v in term_counts.items()])
                            
                            # Display based on selected visualization type
                            if viz_type == "Bar Chart":
                                # Take top 15 for bar chart
                                top_term_df = term_df.head(15)
                                
                                # Special handling for BIRADS to make sure they're in ascending order
                                if category == "birads":
                                    # Sort BIRADS in proper numeric order
                                    try:
                                        # Convert to numeric when possible, otherwise keep as string
                                        term_df['SortOrder'] = term_df['Term'].apply(lambda x: float(x[0]) if x[0].isdigit() else 999)
                                        top_term_df = term_df.sort_values('SortOrder').head(15)
                                        title = "BI-RADS Score Distribution"
                                    except:
                                        title = f"Most Common {category.title()} Terms"
                                else:
                                    title = f"Most Common {category.title()} Terms"
                                    
                                fig = px.bar(top_term_df, x="Term", y="Count", text="Count",
                                            title=title)
                                fig.update_layout(xaxis_title="Term", yaxis_title="Frequency")
                                st.plotly_chart(fig, use_container_width=True)
                            
                            elif viz_type == "Word Cloud":
                                # Create word cloud
                                try:
                                    wordcloud = WordCloud(
                                        width=800, height=400,
                                        background_color='white',
                                        max_words=100,
                                        colormap='viridis'
                                    ).generate_from_frequencies({k: v for k, v in term_counts.items()})
                                    
                                    fig, ax = plt.subplots(figsize=(10, 5))
                                    ax.imshow(wordcloud, interpolation='bilinear')
                                    ax.axis('off')
                                    st.pyplot(fig)
                                except Exception as e:
                                    st.warning(f"Could not generate word cloud: {str(e)}")
                                    # Fall back to bar chart
                                    top_term_df = term_df.head(15)
                                    fig = px.bar(top_term_df, x="Term", y="Count", text="Count")
                                    st.plotly_chart(fig, use_container_width=True)
                            
                            else:  # Table view
                                st.dataframe(term_df, use_container_width=True)
                            
                            # Show term explanations
                            try:
                                explanations = get_term_explanations(category)
                                
                                if explanations:
                                    # For BIRADS, always expand by default
                                    with st.expander("Term Explanations", expanded=(category == 'birads')):
                                        for term, count in term_counts.items():
                                            if term in explanations:
                                                st.markdown(f"**{term}** ({count} occurrences): {explanations[term]}")
                                            else:
                                                st.markdown(f"**{term}** ({count} occurrences)")
                            except Exception as e:
                                st.warning(f"Could not load term explanations: {str(e)}")
                else:
                    st.warning("No text data available for terminology analysis")
            except Exception as e:
                st.error(f"Error analyzing medical terminology: {e}")
                logger.exception("Error in terminology analysis")
        
        with analysis_tab4:
            st.subheader("AI Report Quality Analysis")
            
            # Check if we have reports to analyze
            if 'raw_ocr_text' in df.columns:
                # Option to select specific report
                report_idx = st.selectbox(
                    "Select a report to analyze:",
                    options=list(range(len(df))),
                    format_func=lambda x: f"Report #{x+1}: {df.iloc[x].get('filename', f'Document {x+1}')}"
                )
                
                # Get the selected report text
                report_text = df.iloc[report_idx]['raw_ocr_text']
                
                # Prepare extracted fields if available
                extracted_fields = {}
                for field in ['birads_score', 'recommendation', 'findings', 'impression']:
                    if field in df.columns:
                        extracted_fields[field] = df.iloc[report_idx][field]
                
                # Run the report quality assessment
                quality_assessment = assess_report_quality(report_text, extracted_fields)
                
                # Display overall score with a large metric
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.metric(
                        "Overall Quality Score", 
                        f"{quality_assessment['overall_score']}/100",
                        delta=None
                    )
                    st.markdown(f"**Assessment:** {quality_assessment['assessment']}")
                
                # Display radar chart for the four metrics
                with col2:
                    # Create radar chart data
                    categories = ['Completeness', 'Descriptiveness', 'Structure', 'Critical Elements']
                    values = [
                        quality_assessment['completeness'],
                        quality_assessment['descriptiveness'],
                        quality_assessment['structure'],
                        quality_assessment['critical_elements']
                    ]
                    
                    # Create figure with plotly
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatterpolar(
                        r=values,
                        theta=categories,
                        fill='toself',
                        name='Report Quality'
                    ))
                    
                    fig.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, 100]
                            )
                        ),
                        title="Report Quality Metrics",
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Display issues and recommendations
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Issues Identified")
                    if quality_assessment['issues']:
                        for issue in quality_assessment['issues']:
                            st.markdown(f"- {issue}")
                    else:
                        st.success("No issues identified!")
                
                with col2:
                    st.subheader("Recommendations")
                    if quality_assessment['recommendations']:
                        for rec in quality_assessment['recommendations']:
                            st.markdown(f"- {rec}")
                    else:
                        st.success("No recommendations needed!")
                
                # Display the report text for reference
                with st.expander("View Report Text", expanded=False):
                    st.text_area("", report_text, height=300, key=f"report_text_{report_idx}")
                
                # Batch analysis section
                st.subheader("Batch Quality Analysis")
                
                # Analyze all reports
                if st.button("Analyze All Reports"):
                    with st.spinner("Analyzing report quality across all documents..."):
                        # Collect quality scores for all reports
                        all_scores = []
                        for i in range(len(df)):
                            text = df.iloc[i]['raw_ocr_text']
                            fields = {}
                            for field in ['birads_score', 'recommendation', 'findings', 'impression', 'interpreting_provider']:
                                if field in df.columns:
                                    fields[field] = df.iloc[i][field]
                            
                            assessment = assess_report_quality(text, fields)
                            report_data = {
                                'report_num': i+1,
                                'filename': df.iloc[i].get('filename', f'Document {i+1}'),
                                'interpreting_provider': fields.get('interpreting_provider', 'Unknown Provider'),
                                'overall_score': assessment['overall_score'],
                                'completeness': assessment['completeness'],
                                'descriptiveness': assessment['descriptiveness'],
                                'structure': assessment['structure'],
                                'critical_elements': assessment['critical_elements'],
                                'has_birads': assessment['has_birads'],
                                'has_measurements': assessment['has_measurements'],
                                'has_recommendations': assessment['has_recommendations'],
                                'issues': assessment['issues']
                            }
                            all_scores.append(report_data)
                        
                        # Create a DataFrame for visualization
                        scores_df = pd.DataFrame([{k: v for k, v in score.items() if k != 'issues'} for score in all_scores])
                        
                        # Display the scores
                        st.dataframe(scores_df)
                        
                        # Create a bar chart of overall scores
                        fig = px.bar(
                            scores_df, 
                            x='report_num', 
                            y='overall_score',
                            title="Overall Quality Scores",
                            labels={'report_num': 'Report Number', 'overall_score': 'Quality Score'},
                            color='overall_score',
                            color_continuous_scale='RdYlGn'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show average scores
                        avg_scores = {
                            'Overall': scores_df['overall_score'].mean(),
                            'Completeness': scores_df['completeness'].mean(),
                            'Descriptiveness': scores_df['descriptiveness'].mean(),
                            'Structure': scores_df['structure'].mean(),
                            'Critical Elements': scores_df['critical_elements'].mean()
                        }
                        
                        # Display metrics in columns
                        cols = st.columns(len(avg_scores))
                        for i, (label, value) in enumerate(avg_scores.items()):
                            cols[i].metric(label, f"{value:.1f}")
                        
                        # Provider Quality Analysis
                        st.subheader("Provider Quality Analysis")
                        
                        # Check if we have provider information
                        if 'interpreting_provider' in scores_df.columns:
                            # Analyze provider quality
                            provider_analysis = analyze_provider_quality(all_scores)
                            
                            if provider_analysis:
                                # Create tabs for different providers
                                provider_tabs = st.tabs(list(provider_analysis.keys()))
                                
                                # Show data for each provider
                                for i, (provider, data) in enumerate(provider_analysis.items()):
                                    with provider_tabs[i]:
                                        # Provider summary
                                        col1, col2, col3 = st.columns([1, 1, 2])
                                        
                                        with col1:
                                            st.metric("Quality Score", f"{data['avg_quality']}/100")
                                            st.metric("Report Count", data['report_count'])
                                            st.markdown(f"**Grade: {data['grade']}**")
                                            
                                        with col2:
                                            # Performance relative to average
                                            delta_color = "normal" if data['relative_performance'] == 0 else "inverse" if data['relative_performance'] < 0 else "normal"
                                            st.metric("vs. Average", f"{data['relative_performance']:+.1f}", delta=None)
                                            
                                            # BI-RADS usage
                                            st.metric("BI-RADS Usage", f"{data['birads_usage']['percentage']}%")
                                            
                                            # Recommendations usage
                                            st.metric("Recommendations", f"{data['recommendations_usage']['percentage']}%")
                                        
                                        with col3:
                                            # Radar chart for quality metrics
                                            categories = ['Completeness', 'Descriptiveness', 'Structure', 'Critical Elements']
                                            values = [
                                                data['avg_completeness'],
                                                data['avg_descriptiveness'],
                                                data['avg_structure'],
                                                data['avg_critical_elements']
                                            ]
                                            
                                            fig = go.Figure()
                                            
                                            fig.add_trace(go.Scatterpolar(
                                                r=values,
                                                theta=categories,
                                                fill='toself',
                                                name=provider
                                            ))
                                            
                                            fig.update_layout(
                                                polar=dict(
                                                    radialaxis=dict(
                                                        visible=True,
                                                        range=[0, 100]
                                                    )
                                                ),
                                                title=f"Quality Profile: {provider}",
                                                showlegend=False
                                            )
                                            
                                            st.plotly_chart(fig, use_container_width=True)
                                        
                                        # Top issues
                                        st.subheader("Common Issues")
                                        if data['top_issues']:
                                            issues_df = pd.DataFrame([
                                                {"Issue": issue, "Count": count} 
                                                for issue, count in data['top_issues'].items()
                                            ])
                                            
                                            # Bar chart of issues
                                            fig = px.bar(
                                                issues_df,
                                                x="Issue",
                                                y="Count",
                                                title="Most Common Issues",
                                                color="Count",
                                                color_continuous_scale="Reds"
                                            )
                                            st.plotly_chart(fig, use_container_width=True)
                                        else:
                                            st.info("No common issues found.")
                                            
                                # Provider comparison
                                st.subheader("Provider Comparison")
                                
                                # Create comparison dataframe
                                comparison_data = []
                                for provider, data in provider_analysis.items():
                                    comparison_data.append({
                                        "Provider": provider,
                                        "Quality Score": data['avg_quality'],
                                        "Grade": data['grade'],
                                        "Reports": data['report_count'],
                                        "BI-RADS Usage": f"{data['birads_usage']['percentage']}%",
                                        "Recommendations": f"{data['recommendations_usage']['percentage']}%",
                                        "vs. Average": data['relative_performance']
                                    })
                                
                                comparison_df = pd.DataFrame(comparison_data)
                                st.dataframe(comparison_df)
                                
                                # Bar chart comparing providers
                                fig = px.bar(
                                    comparison_df,
                                    x="Provider",
                                    y="Quality Score",
                                    color="Grade",
                                    title="Provider Quality Comparison",
                                    color_discrete_map={
                                        "A": "#2ecc71",  # Green
                                        "B": "#3498db",  # Blue
                                        "C": "#f1c40f",  # Yellow
                                        "D": "#e67e22",  # Orange
                                        "F": "#e74c3c"   # Red
                                    }
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.info("No provider data available for analysis.")
                        else:
                            st.info("Provider information not available in the dataset.")
            else:
                st.warning("No report text available for quality analysis. Please upload documents first.")
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