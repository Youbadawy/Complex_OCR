"""
UI components for Medical Report Processor application.
Contains reusable UI elements and rendering functions.
"""
import streamlit as st
import logging
import platform
import sys
import tempfile
import os
import pandas as pd
import importlib.metadata as pkg_metadata
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Union
from io import BytesIO
from PIL import Image

# Set up logging
logger = logging.getLogger(__name__)

def render_message(role, content, is_thinking=False):
    """
    Render a chat message in Streamlit
    
    Args:
        role: Message role ("user" or "assistant")
        content: Message content
        is_thinking: Whether to show a thinking status
    """
    if is_thinking:
        with st.chat_message(role):
            with st.status("Thinking...", expanded=True) as status:
                st.markdown(content)
                status.update(label="Done!", state="complete", expanded=False)
    else:
        with st.chat_message(role):
            st.markdown(content)

def check_dependencies(display_in_sidebar=False):
    """
    Check and display the status of key dependencies
    
    Args:
        display_in_sidebar: Whether to display in sidebar
        
    Returns:
        Dict of dependency status information
    """
    # Define key packages to check
    key_packages = [
        "streamlit", 
        "pandas",
        "numpy",
        "transformers",
        "torch",
        "PyMuPDF",
        "paddleocr",
        "matplotlib",
        "plotly",
        "anthropic",
        "scikit-learn",
        "sqlalchemy",
        "nltk"
    ]
    
    # Get system information
    system_info = {
        "python_version": sys.version.split(' ')[0],
        "platform": platform.system(),
        "platform_version": platform.version(),
        "processor": platform.processor()
    }
    
    # Check package versions
    package_info = []
    all_dependencies_available = True
    
    for package in key_packages:
        try:
            version = pkg_metadata.version(package)
            package_info.append({"Package": package, "Version": version, "Status": "✅ Installed"})
        except pkg_metadata.PackageNotFoundError:
            package_info.append({"Package": package, "Version": "Not found", "Status": "❌ Missing"})
            all_dependencies_available = False
    
    # Display info if requested
    if display_in_sidebar:
        with st.sidebar.expander("System Information", expanded=False):
            st.write(f"Python version: {system_info['python_version']}")
            st.write(f"Operating System: {system_info['platform']} {system_info['platform_version']}")
            
            # Display package versions
            st.subheader("Package Versions")
            df = pd.DataFrame(package_info)
            st.dataframe(df, hide_index=True)
    else:
        with st.expander("System Information", expanded=False):
            st.write(f"Python version: {system_info['python_version']}")
            st.write(f"Operating System: {system_info['platform']} {system_info['platform_version']}")
            
            # Display package versions
            st.subheader("Package Versions")
            df = pd.DataFrame(package_info)
            st.dataframe(df, hide_index=True)
    
    return {
        "system_info": system_info,
        "package_info": package_info,
        "all_dependencies_available": all_dependencies_available
    }

def setup_debug_logging():
    """
    Configure detailed debug logging
    """
    # Create a temp file for logging
    log_file = tempfile.NamedTemporaryFile(delete=False, suffix='.log')
    
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file.name),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    st.success(f"Debug logging enabled to {log_file.name}")
    st.session_state.debug_log_path = log_file.name
    
    return log_file.name

def display_report_card(report_data):
    """
    Display a card with key medical report information
    
    Args:
        report_data: Dictionary with report information
    """
    with st.container():
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.subheader(report_data.get('filename', 'Unknown Report'))
            
            # Report date
            if 'exam_date' in report_data and report_data['exam_date']:
                st.write(f"**Exam Date:** {report_data['exam_date']}")
                
            # Report type
            if 'exam_type' in report_data and report_data['exam_type']:
                st.write(f"**Exam Type:** {report_data['exam_type']}")
                
            # Patient info
            if 'patient_name' in report_data and report_data['patient_name']:
                st.write(f"**Patient:** {report_data['patient_name']}")
        
        with col2:
            # BIRADS score if available
            if 'birads_score' in report_data and report_data['birads_score']:
                score = report_data['birads_score']
                color = "#ff4b4b" if score in ['4', '5', '6'] else "#4bc467"
                st.markdown(f"""
                <div style="background-color: {color}; padding: 10px; border-radius: 5px; text-align: center; color: white;">
                    <h3 style="margin: 0;">BIRADS {score}</h3>
                </div>
                """, unsafe_allow_html=True)
        
        # View details button
        if st.button(f"View Details for {report_data.get('filename', 'Report')}"):
            st.session_state.selected_report = report_data

def display_pdf_image(image_data):
    """
    Display PDF images in Streamlit
    
    Args:
        image_data: Image data (PIL Image or bytes)
    """
    if not image_data:
        st.warning("No image data to display")
        return
    
    # If it's a single image, convert to list
    if not isinstance(image_data, list):
        image_data = [image_data]
    
    # Display each image
    for i, img in enumerate(image_data):
        try:
            if isinstance(img, bytes):
                img = Image.open(BytesIO(img))
            
            st.image(img, caption=f"Page {i+1}", use_column_width=True)
        except Exception as e:
            st.error(f"Error displaying image {i+1}: {e}")

def display_file_uploader(accept_multiple=False, file_types=["pdf"]):
    """
    Display a file uploader with consistent styling
    
    Args:
        accept_multiple: Whether to accept multiple files
        file_types: List of accepted file types
        
    Returns:
        Uploaded file(s)
    """
    # Convert file types to proper format
    types = []
    for ftype in file_types:
        if not ftype.startswith('.'):
            types.append('.' + ftype)
        else:
            types.append(ftype)
    
    # Create and return the file uploader
    return st.file_uploader(
        "Choose file(s)" if accept_multiple else "Choose a file",
        accept_multiple_files=accept_multiple,
        type=types
    )

def display_metrics(metrics):
    """
    Display a set of metrics in a row
    
    Args:
        metrics: Dictionary of metric names and values
    """
    # Create columns based on the number of metrics
    cols = st.columns(len(metrics))
    
    # Display each metric
    for i, (label, value) in enumerate(metrics.items()):
        with cols[i]:
            st.metric(label=label, value=value)

def plot_data_overview(df):
    """
    Create visual overview of dataframe data
    
    Args:
        df: Pandas DataFrame with medical report data
    """
    if df is None or df.empty:
        st.warning("No data available for visualization")
        return
    
    st.subheader("Data Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Show exam type distribution if available
        if "Examination Type" in df.columns and not df["Examination Type"].isna().all():
            fig, ax = plt.subplots(figsize=(8, 6))
            df["Examination Type"].fillna("Unknown").value_counts().plot(
                kind='pie', autopct='%1.1f%%', ax=ax
            )
            plt.title("Examination Types")
            plt.ylabel("")
            plt.tight_layout()
            st.pyplot(fig)
    
    with col2:
        # Show BIRADS distribution if available
        if "BIRADS Score" in df.columns and not df["BIRADS Score"].isna().all():
            fig, ax = plt.subplots(figsize=(8, 6))
            df["BIRADS Score"].fillna("Unknown").value_counts().sort_index().plot(
                kind='bar', ax=ax
            )
            plt.title("BIRADS Score Distribution")
            plt.xlabel("BIRADS Score")
            plt.ylabel("Count")
            plt.tight_layout()
            st.pyplot(fig)
    
    # Show data table
    st.subheader("Data Table")
    st.dataframe(df)
    
    # Download button
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "Download as CSV",
        csv,
        "report_data.csv",
        "text/csv",
        key='download-csv'
    ) 