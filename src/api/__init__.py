"""
API module for Medical Report Processor application.
This package contains API client implementations.
"""

from .claude_client import initialize_claude, get_claude_response, claude_client

__all__ = [
    'initialize_claude',
    'get_claude_response',
    'claude_client'
] 