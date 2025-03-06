"""
UI module for Medical Report Processor application.
This package contains UI components and tab rendering functions.
"""

from .components import (
    render_message, display_report_card, display_pdf_image, 
    display_file_uploader, check_dependencies, setup_debug_logging,
    display_metrics, plot_data_overview
)
from .tabs import (
    render_upload_tab, render_analysis_tab, 
    render_chatbot_tab, render_database_tab
)

__all__ = [
    'render_message',
    'display_report_card',
    'display_pdf_image',
    'display_file_uploader',
    'check_dependencies',
    'setup_debug_logging',
    'display_metrics',
    'plot_data_overview',
    'render_upload_tab',
    'render_analysis_tab',
    'render_chatbot_tab',
    'render_database_tab'
] 