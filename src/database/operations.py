"""
Database operations for Medical Report Processor application.
"""
import json
import logging
import datetime
from typing import Dict, List, Any, Optional, Union
from sqlalchemy import create_engine, inspect, Column, String, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Set up logging
logger = logging.getLogger(__name__)

# SQLAlchemy setup
Base = declarative_base()
engine = create_engine('sqlite:///medical_reports.db')
Session = sessionmaker(bind=engine)

# Define Report model
class Report(Base):
    __tablename__ = 'reports'
    
    md5_hash = Column(String, primary_key=True)
    filename = Column(String)
    processed_at = Column(DateTime, default=datetime.datetime.now)
    patient_name = Column(String)
    exam_date = Column(String)
    findings = Column(Text)
    report_metadata = Column(Text)
    raw_ocr_text = Column(Text)
    
    def __repr__(self):
        return f"<Report(filename='{self.filename}', processed_at='{self.processed_at}')>"

def init_db():
    """
    Initialize the database
    
    Returns:
        bool: True if successful
    """
    try:
        Base.metadata.create_all(engine)
        logger.info("Database initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        return False

def save_to_db(data: dict):
    """
    Save report data to database
    
    Args:
        data: Report data dictionary
        
    Returns:
        bool: True if successful
    """
    session = None
    try:
        session = Session()
        
        # Ensure exam_date is properly serialized
        exam_date = data.get('exam_date', 'Not Available')
        if isinstance(exam_date, dict) and 'value' in exam_date:
            exam_date = exam_date.get('value', 'Not Available')
        elif hasattr(exam_date, 'value'):
            exam_date = exam_date.value if exam_date.value != "N/A" else 'Not Available'
        
        # Ensure findings are properly serialized
        findings = data.get('findings', {})
        # Convert any ExtractedField objects within findings
        if isinstance(findings, dict):
            for key, value in list(findings.items()):
                if hasattr(value, 'value'):
                    findings[key] = value.value
                elif isinstance(value, dict) and 'value' in value:
                    findings[key] = value['value']
        
        # Convert dict fields to JSON
        report = Report(
            md5_hash=data['md5_hash'],
            filename=data['filename'],
            patient_name=data.get('patient_name', 'Not Available'),
            exam_date=str(exam_date),
            findings=json.dumps(findings),
            report_metadata=json.dumps(data.get('metadata', {})),
            raw_ocr_text=data.get('raw_ocr_text', '')
        )
        
        # Add and commit
        session.merge(report)  # Use merge instead of add to handle updates
        session.commit()
        logger.info(f"Saved report to database: {data['filename']}")
        return True
        
    except Exception as e:
        logger.error(f"Database error: {str(e)}")
        if session:
            session.rollback()
        return False
    finally:
        if session:
            session.close()

def get_all_reports():
    """
    Get all reports from database
    
    Returns:
        List[Report]: List of Report objects
    """
    session = None
    try:
        session = Session()
        reports = session.query(Report).all()
        return reports
    except Exception as e:
        logger.error(f"Error getting reports: {e}")
        return []
    finally:
        if session:
            session.close()

def get_report_by_hash(md5_hash):
    """
    Get report by md5 hash
    
    Args:
        md5_hash: MD5 hash of report
        
    Returns:
        Optional[Report]: Report object if found, None otherwise
    """
    session = None
    try:
        session = Session()
        report = session.query(Report).filter_by(md5_hash=md5_hash).first()
        return report
    except Exception as e:
        logger.error(f"Error getting report {md5_hash}: {e}")
        return None
    finally:
        if session:
            session.close()

def delete_report(md5_hash):
    """
    Delete report by md5 hash
    
    Args:
        md5_hash: MD5 hash of report
        
    Returns:
        bool: True if successful
    """
    session = None
    try:
        session = Session()
        report = session.query(Report).filter_by(md5_hash=md5_hash).first()
        if report:
            session.delete(report)
            session.commit()
            logger.info(f"Deleted report: {report.filename}")
            return True
        else:
            logger.warning(f"Report not found: {md5_hash}")
            return False
    except Exception as e:
        logger.error(f"Error deleting report {md5_hash}: {e}")
        if session:
            session.rollback()
        return False
    finally:
        if session:
            session.close()

def migrate_database_schema():
    """
    Migrate database schema to latest version
    
    Returns:
        Dict[str, Any]: Status of migration
    """
    status = {"success": True, "messages": []}
    
    try:
        # Get inspector
        inspector = inspect(engine)
        
        # Check if table exists
        if not inspector.has_table('reports'):
            Base.metadata.create_all(engine)
            status["messages"].append("Created reports table")
        
        # Check for missing columns
        missing_columns = []
        existing_columns = [col['name'] for col in inspector.get_columns('reports')]
        
        expected_columns = [
            'md5_hash', 'filename', 'processed_at', 'patient_name', 
            'exam_date', 'findings', 'report_metadata', 'raw_ocr_text'
        ]
        
        for col in expected_columns:
            if col not in existing_columns:
                missing_columns.append(col)
        
        if missing_columns:
            status["messages"].append(f"Missing columns: {', '.join(missing_columns)}")
            status["success"] = False
        else:
            status["messages"].append("Database schema is up to date")
        
    except Exception as e:
        status["success"] = False
        status["messages"].append(f"Error checking database: {str(e)}")
        logger.exception("Error in migrate_database_schema")
    
    return status 