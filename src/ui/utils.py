"""
UI utility functions for Medical Report Processor application.
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
from typing import List, Any, Dict, Union, Optional
from PIL import Image
import platform
import importlib.metadata as pkg_metadata
import sys

# Try importing PIL, but provide fallback if not available
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    import logging
    logging.warning("PIL not available. Image display will be limited.")

def display_pdf_image(image_data: Union[bytes, List[bytes], Image.Image, List[Image.Image]]):
    """
    Display PDF images in Streamlit
    
    Args:
        image_data: PDF image data as bytes, PIL Image, or lists of either
    """
    if not image_data:
        st.warning("No image data to display")
        return
    
    # Handle single image
    if not isinstance(image_data, list):
        image_data = [image_data]
    
    # Display each image
    for i, img in enumerate(image_data):
        try:
            # Convert to PIL Image if needed
            if isinstance(img, bytes):
                img = Image.open(io.BytesIO(img))
                
            # Display the image
            st.image(img, caption=f"Page {i+1}", use_column_width=True)
            
        except Exception as e:
            st.error(f"Error displaying image {i+1}: {e}")

def check_dependencies(display_in_sidebar=False):
    """
    Check and display the status of key dependencies
    
    Args:
        display_in_sidebar: Whether to display in sidebar (True) or main area (False)
    """
    # Define key packages to check
    key_packages = [
        "streamlit", 
        "pandas",
        "numpy",
        "transformers",
        "torch",
        "PyMuPDF",
        "matplotlib",
        "plotly",
        "anthropic",
        "scikit-learn"
    ]
    
    # Get the display function based on location
    display_func = st.sidebar if display_in_sidebar else st
    
    with display_func.expander("System Information", expanded=False):
        # System info
        st.write(f"Python version: {sys.version.split(' ')[0]}")
        st.write(f"Operating System: {platform.system()} {platform.version()}")
        
        # Display package versions
        st.subheader("Package Versions")
        package_info = []
        
        for package in key_packages:
            try:
                version = pkg_metadata.version(package)
                package_info.append({"Package": package, "Version": version, "Status": "✅ Installed"})
            except pkg_metadata.PackageNotFoundError:
                package_info.append({"Package": package, "Version": "Not found", "Status": "❌ Missing"})
        
        # Display as a dataframe
        df = pd.DataFrame(package_info)
        st.dataframe(df, hide_index=True)

def render_metric_cards(metrics: Dict[str, Any]):
    """
    Render a set of metric cards in a grid layout
    
    Args:
        metrics: Dictionary of metric name to value pairs
    """
    # Calculate number of columns (max 3)
    num_metrics = len(metrics)
    num_cols = min(3, num_metrics)
    
    # Create columns
    cols = st.columns(num_cols)
    
    # Distribute metrics across columns
    for i, (label, value) in enumerate(metrics.items()):
        col_idx = i % num_cols
        with cols[col_idx]:
            st.metric(
                label=label,
                value=value
            )

def plot_bar_chart(data: Dict[str, Any], title: str, x_label: str, y_label: str):
    """
    Create and display a bar chart
    
    Args:
        data: Dictionary mapping categories to values
        title: Chart title
        x_label: X-axis label
        y_label: Y-axis label
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot the data
    categories = list(data.keys())
    values = list(data.values())
    
    bars = ax.bar(categories, values)
    
    # Add labels and title
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                str(height), ha='center', va='bottom')
    
    # Rotate x-labels if there are many categories
    if len(categories) > 5:
        plt.xticks(rotation=45, ha='right')
    
    # Adjust layout
    plt.tight_layout()
    
    # Display in Streamlit
    st.pyplot(fig)

def plot_pie_chart(data: Dict[str, Any], title: str):
    """
    Create and display a pie chart
    
    Args:
        data: Dictionary mapping categories to values
        title: Chart title
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot the data
    categories = list(data.keys())
    values = list(data.values())
    
    # Only include non-zero values
    filtered_categories = []
    filtered_values = []
    for cat, val in zip(categories, values):
        if val > 0:
            filtered_categories.append(cat)
            filtered_values.append(val)
    
    ax.pie(
        filtered_values, 
        labels=filtered_categories, 
        autopct='%1.1f%%',
        startangle=90
    )
    
    # Equal aspect ratio ensures that pie is drawn as a circle
    ax.axis('equal')
    ax.set_title(title)
    
    # Display in Streamlit
    st.pyplot(fig)

def display_text_with_highlights(text: str, highlights: List[str], case_sensitive: bool = False):
    """
    Display text with highlighted terms
    
    Args:
        text: The text to display
        highlights: List of terms to highlight
        case_sensitive: Whether the highlighting should be case sensitive
    """
    import re
    
    # Create a copy of the text for highlighting
    highlighted_text = text
    
    # Sort highlights by length (descending) to handle overlapping terms
    highlights = sorted(highlights, key=len, reverse=True)
    
    # Define the highlight style
    highlight_style = '<span style="background-color: #FFFF00;">{}</span>'
    
    # Apply highlights
    for term in highlights:
        if not term.strip():
            continue
            
        # Create pattern based on case sensitivity
        if case_sensitive:
            pattern = re.escape(term)
        else:
            pattern = re.escape(term)
            flags = re.IGNORECASE
            
        # Replace all occurrences with highlighted version
        if case_sensitive:
            highlighted_text = re.sub(
                pattern, 
                lambda m: highlight_style.format(m.group(0)), 
                highlighted_text
            )
        else:
            highlighted_text = re.sub(
                pattern, 
                lambda m: highlight_style.format(m.group(0)), 
                highlighted_text, 
                flags=flags
            )
    
    # Display with Streamlit markdown
    st.markdown(highlighted_text, unsafe_allow_html=True)

def format_date(date_str: str) -> str:
    """
    Format a date string for display
    
    Args:
        date_str: ISO format date string
        
    Returns:
        Formatted date string
    """
    from datetime import datetime
    
    try:
        dt = datetime.fromisoformat(date_str)
        return dt.strftime("%B %d, %Y, %I:%M %p")
    except:
        return date_str

def format_chat_message(role: str, content: str, is_thinking: bool = False) -> None:
    """
    Format a chat message in Streamlit
    
    Args:
        role: Message role ("user" or "assistant")
        content: Message content
        is_thinking: Whether this is a thinking message
    """
    if is_thinking:
        with st.chat_message(role):
            with st.status("Thinking...", expanded=True) as status:
                st.markdown(content)
                status.update(label="Done!", state="complete", expanded=False)
    else:
        with st.chat_message(role):
            st.markdown(content)

def render_dataframe_download(df: pd.DataFrame, filename: str = "download.csv") -> None:
    """
    Render a download button for a dataframe
    
    Args:
        df: Dataframe to download
        filename: Download filename
    """
    if df is None or df.empty:
        st.warning("No data available for download")
        return
    
    # Convert to CSV
    csv = df.to_csv(index=False)
    
    # Add download button
    st.download_button(
        label="Download as CSV",
        data=csv,
        file_name=filename,
        mime="text/csv"
    )

def render_json_download(data: Dict[str, Any], filename: str = "download.json") -> None:
    """
    Render a download button for JSON data
    
    Args:
        data: Dictionary to download as JSON
        filename: Download filename
    """
    if not data:
        st.warning("No data available for download")
        return
    
    import json
    
    # Convert to JSON string
    json_str = json.dumps(data, indent=2)
    
    # Add download button
    st.download_button(
        label="Download as JSON",
        data=json_str,
        file_name=filename,
        mime="application/json"
    )

def create_figure_for_streamlit(figsize: tuple = (10, 6)) -> plt.Figure:
    """
    Create a matplotlib figure with proper sizing for Streamlit
    
    Args:
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax 