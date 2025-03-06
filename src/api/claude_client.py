"""
Claude API integration for Medical Report Processor application.
"""
import os
import logging
import streamlit as st
from typing import Dict, Any, Optional, List

# Set up logging
logger = logging.getLogger(__name__)

# Try to import the Anthropic library
try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    logger.warning("Anthropic library not available. Claude features will be disabled.")

# Global Claude client
claude_client = None

def initialize_claude():
    """
    Initialize the Claude API client by checking for API key and setting up the client.
    
    Returns:
        bool: True if successful, False otherwise
    """
    global claude_client
    
    if not ANTHROPIC_AVAILABLE:
        logger.warning("Anthropic library not available. Claude features will be disabled.")
        return False
    
    # Check if API key is set
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.warning("ANTHROPIC_API_KEY environment variable not set. Claude features will be disabled.")
        return False
    
    try:
        # Create API client
        claude_client = Anthropic(api_key=api_key)
        logger.info("Claude API client initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Error initializing Claude API client: {e}")
        claude_client = None
        return False

def get_claude_response(user_query: str, chat_history: Optional[List[Dict[str, str]]] = None, 
                       system_prompt: Optional[str] = None):
    """
    Get a response from the Claude API
    
    Args:
        user_query: User's question or message
        chat_history: Optional list of previous chat messages
        system_prompt: Optional system prompt for context
        
    Returns:
        str: Claude's response
    """
    global claude_client
    
    if not claude_client:
        logger.warning("Claude API client not initialized")
        return "Claude API is not available. Please check your API key and try again."
    
    try:
        # Use the model selected in session state or default
        model = st.session_state.get('selected_claude_model', "claude-3-haiku-20240307")
        
        # Set default system prompt if not provided
        if system_prompt is None:
            system_prompt = """
            You are a helpful medical assistant analyzing medical reports and documents.
            Provide factual, accurate information based on the content provided.
            If you're unsure about something, admit your limitations rather than making up information.
            """
        
        # Prepare messages
        messages = []
        
        # Add chat history if provided
        if chat_history:
            for message in chat_history:
                messages.append({
                    "role": message["role"],
                    "content": message["content"]
                })
        
        # Add user query
        messages.append({"role": "user", "content": user_query})
        
        # Call Claude API
        response = claude_client.messages.create(
            model=model,
            max_tokens=1000,
            system=system_prompt,
            messages=messages
        )
        
        # Extract and return the response content
        if response and response.content and len(response.content) > 0:
            return response.content[0].text
        else:
            return "I couldn't generate a response at this time. Please try again."
        
    except Exception as e:
        logger.error(f"Error calling Claude API: {e}")
        return f"Error getting response from Claude: {str(e)}" 