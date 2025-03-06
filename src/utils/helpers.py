"""
Utility helper functions for Medical Report Processor application.
"""
import streamlit as st
import logging
import os
import hashlib
import tempfile
from typing import Dict, Any, Optional, List
import json
import re
from pathlib import Path

# Set up logging
logger = logging.getLogger(__name__)

def clear_all_caches():
    """
    Clear all application caches
    
    Returns:
        bool: True if successful
    """
    # Clear Streamlit caches
    st.cache_data.clear()
    st.cache_resource.clear()
    
    # Clear session state caches
    for key in list(st.session_state.keys()):
        if key.endswith('_cache') or key in [
            'single_result', 'batch_results', 'selected_report', 
            'ocr_results', 'structured_data', 'text'
        ]:
            del st.session_state[key]
    
    logger.info("All caches cleared")
    return True

def fix_database_issues():
    """
    Check and fix common database issues
    
    Returns:
        Dict[str, Any]: Status of fixes applied
    """
    from sqlalchemy import inspect, text
    from database.operations import engine, Base
    
    status = {"success": True, "messages": []}
    
    try:
        # Get inspector
        inspector = inspect(engine)
        
        # Check if reports table exists
        if not inspector.has_table('reports'):
            status["messages"].append("Reports table does not exist. Creating table...")
            Base.metadata.create_all(engine)
            status["messages"].append("Created reports table")
        else:
            status["messages"].append("Reports table exists")
        
        # Check for missing columns and add them if needed
        required_columns = {
            'md5_hash': 'VARCHAR', 
            'filename': 'VARCHAR',
            'processed_at': 'DATETIME',
            'patient_name': 'VARCHAR',
            'exam_date': 'VARCHAR',
            'findings': 'TEXT',
            'report_metadata': 'TEXT',
            'raw_ocr_text': 'TEXT'
        }
        
        existing_columns = {col['name']: col['type'] for col in inspector.get_columns('reports')}
        
        for col_name, col_type in required_columns.items():
            if col_name not in existing_columns:
                # Add missing column
                sql = f"ALTER TABLE reports ADD COLUMN {col_name} {col_type}"
                engine.execute(text(sql))
                status["messages"].append(f"Added missing column: {col_name}")
        
        # Check for index on md5_hash
        indices = inspector.get_indexes('reports')
        has_md5_index = any('md5_hash' in idx.get('column_names', []) for idx in indices)
        
        if not has_md5_index:
            sql = "CREATE INDEX idx_md5_hash ON reports (md5_hash)"
            engine.execute(text(sql))
            status["messages"].append("Created index on md5_hash column")
        
        # Vacuum the database
        engine.execute(text("VACUUM"))
        status["messages"].append("Vacuumed database")
        
    except Exception as e:
        status["success"] = False
        status["messages"].append(f"Error fixing database: {str(e)}")
        logger.exception("Error fixing database")
    
    return status

def generate_md5(text: str) -> str:
    """
    Generate MD5 hash from text
    
    Args:
        text: Input text
        
    Returns:
        str: MD5 hash
    """
    return hashlib.md5(text.encode()).hexdigest()

def safe_json_loads(json_str: str, default=None) -> Any:
    """
    Safely load JSON string
    
    Args:
        json_str: JSON string
        default: Default value if parsing fails
        
    Returns:
        Parsed JSON or default value
    """
    if not json_str:
        return default
    
    try:
        return json.loads(json_str)
    except Exception as e:
        logger.warning(f"Error parsing JSON: {e}")
        return default

def create_temp_file(content: bytes, suffix: str = '.tmp') -> str:
    """
    Create a temporary file with given content
    
    Args:
        content: File content as bytes
        suffix: File suffix
        
    Returns:
        str: Path to temporary file
    """
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_file.write(content)
    temp_file.close()
    return temp_file.name

def clean_filename(filename: str) -> str:
    """
    Clean filename to remove unsafe characters
    
    Args:
        filename: Input filename
        
    Returns:
        str: Cleaned filename
    """
    # Remove unsafe characters
    safe_filename = re.sub(r'[^\w\s.-]', '', filename)
    # Replace spaces with underscores
    safe_filename = safe_filename.replace(' ', '_')
    return safe_filename

def ensure_directory(path: str) -> bool:
    """
    Ensure directory exists, creating it if necessary
    
    Args:
        path: Directory path
        
    Returns:
        bool: True if directory exists or was created successfully
    """
    try:
        dir_path = Path(path)
        if not dir_path.exists():
            dir_path.mkdir(parents=True)
            logger.info(f"Created directory: {path}")
        return True
    except Exception as e:
        logger.error(f"Failed to create directory {path}: {e}")
        return False 