"""
Model loading and initialization functions for Medical Report Processor.
"""
import streamlit as st
import logging
import os
from typing import Dict, Any, Optional
import importlib
import sys
import warnings

# Set up logging
logger = logging.getLogger(__name__)

@st.cache_resource(show_spinner=False)
def load_models():
    """
    Load and cache all models used in the application.
    
    Returns:
        Dict[str, Any]: Dictionary of loaded models
    """
    models = {}
    
    # Try to initialize Claude
    try:
        from api.claude_client import initialize_claude
        claude_client = initialize_claude()
        if claude_client:
            models['claude'] = claude_client
            logger.info("Claude API initialized successfully")
        else:
            logger.warning("Claude API initialization failed")
    except Exception as e:
        logger.error(f"Error initializing Claude API: {e}")
    
    # Try to load PyTorch for model inference
    try:
        import torch
        models['torch_available'] = True
        logger.info(f"PyTorch {torch.__version__} loaded successfully")
        
        # Check CUDA availability
        if torch.cuda.is_available():
            models['cuda_available'] = True
            models['cuda_devices'] = torch.cuda.device_count()
            models['cuda_version'] = torch.version.cuda
            logger.info(f"CUDA is available with {models['cuda_devices']} device(s)")
        else:
            models['cuda_available'] = False
            logger.info("CUDA is not available")
    except ImportError:
        models['torch_available'] = False
        models['cuda_available'] = False
        logger.warning("PyTorch is not installed")
    
    # Initialize OCR models
    try:
        from document_processing.ocr import init_paddle
        paddle_initialized = init_paddle()
        models['paddle_ocr'] = paddle_initialized
        logger.info(f"PaddleOCR initialized: {paddle_initialized}")
    except Exception as e:
        models['paddle_ocr'] = False
        logger.warning(f"Failed to initialize PaddleOCR: {e}")
    
    # Initialize Donut model for document understanding
    try:
        donut_model = init_donut()
        if donut_model:
            models['donut'] = donut_model
            logger.info("Donut model loaded successfully")
    except Exception as e:
        logger.warning(f"Failed to load Donut model: {e}")
    
    # Create chatbot function
    if 'claude' in models:
        from api.claude_client import get_claude_response
        def chatbot(prompt):
            return get_claude_response(prompt, models['claude'])
        models['chatbot'] = chatbot
    else:
        models['chatbot'] = chatbot_fallback
    
    # Initialize NLTK resources
    try:
        ensure_nltk_resources()
        models['nltk_available'] = True
        logger.info("NLTK resources loaded successfully")
    except Exception as e:
        models['nltk_available'] = False
        logger.warning(f"Failed to load NLTK resources: {e}")
    
    # Try to load translation model
    try:
        translation_model = load_translation_model()
        if translation_model:
            models['translation'] = translation_model
            logger.info("Translation model loaded successfully")
    except Exception as e:
        logger.warning(f"Failed to load translation model: {e}")
    
    return models

def chatbot_fallback(text):
    """
    Fallback function when chatbot is not available
    
    Args:
        text: Input text from user
        
    Returns:
        str: Response message
    """
    return ("I'm sorry, I don't have access to the Claude API at the moment. "
            "Please ensure you have set up your Anthropic API key correctly "
            "and try again later.")

@st.cache_resource
def load_translation_model():
    """
    Load and cache translation model
    
    Returns:
        Optional[Any]: Translation model if successful, None otherwise
    """
    try:
        from transformers import MarianMTModel, MarianTokenizer
        
        # Load Helsinki-NLP translation model
        model_name = "Helsinki-NLP/opus-mt-en-ROMANCE"
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        
        return {
            'model': model,
            'tokenizer': tokenizer
        }
    except Exception as e:
        logger.warning(f"Failed to load translation model: {e}")
        return None

def ensure_nltk_resources():
    """
    Ensure required NLTK resources are downloaded
    """
    try:
        import nltk
        
        # Resources needed for NLP tasks
        resources = [
            'punkt',      # For sentence tokenization
            'stopwords',  # Common stopwords 
            'wordnet',    # For lemmatization
            'averaged_perceptron_tagger'  # For POS tagging
        ]
        
        # Download resources if not already present
        for resource in resources:
            try:
                nltk.data.find(f'tokenizers/{resource}')
            except LookupError:
                nltk.download(resource, quiet=True)
    except ImportError:
        logger.warning("NLTK is not installed")
        raise

def init_donut():
    """
    Initialize Donut document understanding model
    
    Returns:
        Optional[Any]: Donut processor and model if successful, None otherwise
    """
    try:
        # Attempt to import required libraries
        from transformers import DonutProcessor, VisionEncoderDecoderModel
        import torch
        
        # Check if we have enough resources to load the model
        if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory > 4 * 1024 * 1024 * 1024:
            # We have a GPU with >4GB memory
            device = "cuda"
        else:
            # Use CPU (slow but works)
            device = "cpu" 
            
        # Initialize processor
        processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
        
        # Initialize model
        model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
        
        # Move model to the appropriate device
        model.to(device)
        
        return {
            'processor': processor,
            'model': model,
            'device': device
        }
    except Exception as e:
        logger.warning(f"Failed to initialize Donut model: {e}")
        return None 