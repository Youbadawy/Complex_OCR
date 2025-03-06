"""
Database models for the Medical Report Processor application.
"""
import datetime
from sqlalchemy import Column, String, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Report(Base):
    """
    SQLAlchemy model for storing processed medical reports.
    Each report is uniquely identified by its MD5 hash.
    """
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
        """String representation of the Report object"""
        return f"<Report(md5_hash='{self.md5_hash[:8]}...', filename='{self.filename}')>" 