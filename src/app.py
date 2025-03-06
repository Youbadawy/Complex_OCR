"""
Main application entry point for Medical Report Processor.

This application provides OCR processing and analysis for medical reports,
specifically focused on mammography and breast imaging.
"""

import streamlit as st
import logging
import warnings
import os
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import components from new modular structure
from models.model_loader import load_models
from database.operations import init_db, migrate_database_schema
from ui.components import check_dependencies, setup_debug_logging
from ui.tabs import (
    render_upload_tab, render_analysis_tab, 
    render_chatbot_tab, render_database_tab
)
from utils.helpers import clear_all_caches, fix_database_issues
from api.claude_client import initialize_claude

# Import new parser components for availability checks - BUT DO NOT MODIFY OCR PROCESS
try:
    from document_processing.medical_report_parser import MedicalReportParser
    from document_processing.parser_integration import process_ocr_text
    PARSER_AVAILABLE = True
except ImportError:
    PARSER_AVAILABLE = False
    logger.warning("New medical report parser not available. Using legacy extraction.")

# Set page config
st.set_page_config(
    page_title="Medical Report Processor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    """Main application entry point"""
    # Add app title
    st.sidebar.title("Medical Report Processor")
    st.sidebar.markdown("---")
    
    # Initialize database
    init_db()
    
    # Initialize API clients
    initialize_claude()
    
    # Check for debug mode
    debug_mode = st.sidebar.checkbox("Debug Mode", value=False)
    
    if debug_mode:
        st.sidebar.warning("Debug mode is enabled")
        if st.sidebar.button("Enable Debug Logging"):
            log_path = setup_debug_logging()
            st.sidebar.info(f"Debug logs: {log_path}")
    
    # Load models
    with st.spinner("Loading models..."):
        models = load_models()
    
    # System checks
    with st.sidebar.expander("System Status", expanded=False):
        # Check dependencies
        check_dependencies(display_in_sidebar=True)
        
        # Database status
        st.subheader("Database Status")
        db_status = migrate_database_schema()
        
        if db_status["success"]:
            st.success("Database OK")
        else:
            st.error("Database issues detected")
            
            # Show fix button
            if st.button("Fix Database Issues"):
                fix_status = fix_database_issues()
                
                if fix_status["success"]:
                    st.success("Database fixed")
                    for msg in fix_status["messages"]:
                        st.write(f"- {msg}")
                else:
                    st.error("Failed to fix database")
                    for msg in fix_status["messages"]:
                        st.write(f"- {msg}")
        
        # Cache management
        st.subheader("Cache Management")
        if st.button("Clear All Caches"):
            clear_all_caches()
            st.success("All caches cleared")
    
    # Main tabs - Changed from sidebar radio to main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Upload", "Analysis", "Chat", "Database"])
    
    # Display the content in each tab
    with tab1:
        render_upload_tab(models)
    
    with tab2:
        render_analysis_tab(models)
    
    with tab3:
        render_chatbot_tab(models)
    
    with tab4:
        render_database_tab()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info(
        "Medical Report Processor v1.0\n"
        "A tool for OCR processing and analysis of medical reports."
    )

if __name__ == "__main__":
    main()
