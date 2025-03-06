"""
LLM-based extraction module for Medical Report Processor application.
Provides functionality to enhance extraction using the Groq API with Deepseek Llama 70b.
"""

import os
import json
import logging
import time
from typing import Dict, Any, List, Optional, Union
import requests

# Setup logger
logger = logging.getLogger(__name__)

# Check if LLM integration is available
try:
    # Try to get API key to check if environment is set up
    api_key = os.environ.get("GROQ_API_KEY")
    LLM_AVAILABLE = api_key is not None
    if not LLM_AVAILABLE:
        logger.warning("GROQ_API_KEY environment variable not set. LLM extraction will not be available.")
except Exception as e:
    logger.warning(f"Error checking LLM availability: {str(e)}")
    LLM_AVAILABLE = False

# Default model
DEFAULT_MODEL = "deepseek-llm/deepseek-llama-70b-chat"

class LLMExtractionError(Exception):
    """Base exception for LLM extraction errors"""
    pass

class APIConnectionError(LLMExtractionError):
    """Exception raised when API connection fails"""
    pass

class APIResponseError(LLMExtractionError):
    """Exception raised when API response is invalid"""
    pass

class PromptEngineeringError(LLMExtractionError):
    """Exception raised when prompt engineering fails"""
    pass

def get_api_key() -> str:
    """
    Get Groq API key from environment variable.
    
    Returns:
        API key string
    
    Raises:
        APIConnectionError: If API key is not found
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        logger.error("GROQ_API_KEY environment variable not found")
        raise APIConnectionError("GROQ_API_KEY environment variable not found")
    return api_key

def call_groq_api(
    prompt: str, 
    model: str = DEFAULT_MODEL, 
    temperature: float = 0.2,
    max_tokens: int = 1024,
    retry_attempts: int = 3,
    retry_delay: float = 1.0
) -> str:
    """
    Call Groq API with provided prompt.
    
    Args:
        prompt: Prompt text to send to the LLM
        model: Model identifier string
        temperature: Sampling temperature (0.0 to 1.0)
        max_tokens: Maximum number of tokens to generate
        retry_attempts: Number of retry attempts
        retry_delay: Delay between retries in seconds
        
    Returns:
        Generated text response
        
    Raises:
        APIConnectionError: If connection to API fails
        APIResponseError: If API response is invalid
    """
    try:
        api_key = get_api_key()
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        # Implement retry logic
        for attempt in range(retry_attempts):
            try:
                logger.debug(f"Calling Groq API with model {model}")
                response = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=30  # 30 second timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if "choices" in result and len(result["choices"]) > 0:
                        return result["choices"][0]["message"]["content"]
                    else:
                        logger.error(f"Invalid API response structure: {result}")
                        raise APIResponseError("Invalid API response structure")
                else:
                    logger.warning(f"API call failed with status {response.status_code}: {response.text}")
                    if attempt < retry_attempts - 1:
                        logger.info(f"Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        # Increase delay for next attempt
                        retry_delay *= 2
                    else:
                        logger.error(f"Failed after {retry_attempts} attempts")
                        raise APIConnectionError(f"API call failed with status {response.status_code}: {response.text}")
                        
            except requests.RequestException as e:
                logger.warning(f"Request exception: {str(e)}")
                if attempt < retry_attempts - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    # Increase delay for next attempt
                    retry_delay *= 2
                else:
                    logger.error(f"Failed after {retry_attempts} attempts")
                    raise APIConnectionError(f"Request exception: {str(e)}")
                    
    except Exception as e:
        logger.error(f"Unexpected error in call_groq_api: {str(e)}")
        raise

def create_extraction_prompt(text: str, extraction_type: str = "full") -> str:
    """
    Create a prompt for medical report extraction.
    
    Args:
        text: Raw OCR text from medical report
        extraction_type: Type of extraction ("full", "birads", "impression", etc.)
        
    Returns:
        Formatted prompt string
    """
    if extraction_type == "full":
        prompt = f"""You are an expert medical NLP system specialized in mammogram report analysis.
Extract the following structured information from this mammogram report. 
Return ONLY a valid JSON object with these keys (leave empty strings for missing information):

- patient_name: The patient's name
- exam_date: The date of the examination (YYYY-MM-DD format)
- exam_type: The type of examination (e.g., MAMMOGRAM, ULTRASOUND)
- birads_score: The BIRADS score (e.g., "BIRADS 1", "BIRADS 2")
- findings: The radiologist's observed findings
- impression: The radiologist's conclusion or assessment
- recommendation: Follow-up recommendations
- clinical_history: Patient's medical history relevant to the exam
- provider_info: Dictionary with referring_provider, interpreting_provider, and facility
- signed_by: Name of the signing physician

Medical report:
```
{text}
```

JSON response:"""

    elif extraction_type == "birads":
        prompt = f"""You are an expert medical NLP system specializing in BIRADS classification.
Carefully analyze this mammogram report and extract the BIRADS score.
The score may be written as BIRADS, BI-RADS, BLRADS, ACR Category, or described implicitly.
Consider the entire report context, especially the impression and recommendation sections.

Return ONLY a valid JSON with two fields:
- "value": the BIRADS score in the format "BIRADS X" (where X is 0-6, possibly with a,b,c subdivisions)
- "confidence": a value from 0.0 to 1.0 indicating your confidence in the extraction

Medical report:
```
{text}
```

JSON response:"""

    elif extraction_type == "sections":
        prompt = f"""You are an expert medical NLP system specializing in mammogram report analysis.
Carefully analyze this mammogram report and separate it into its clinical sections.

Return ONLY a valid JSON with these fields (leave empty strings for missing sections):
- "clinical_history": The patient's medical history section
- "findings": The radiologist's findings/observations section
- "impression": The radiologist's assessment/conclusion section
- "recommendation": The follow-up recommendations section

Medical report:
```
{text}
```

JSON response:"""

    else:
        logger.warning(f"Unknown extraction_type: {extraction_type}")
        raise PromptEngineeringError(f"Unknown extraction type: {extraction_type}")
        
    return prompt

def parse_llm_response(response: str) -> Dict[str, Any]:
    """
    Parse LLM response into a structured dictionary.
    
    Args:
        response: Raw LLM response text
        
    Returns:
        Parsed dictionary
        
    Raises:
        APIResponseError: If response cannot be parsed
    """
    try:
        # Extract JSON from response (handling cases where there might be extra text)
        json_start = response.find('{')
        json_end = response.rfind('}') + 1
        
        if json_start == -1 or json_end == 0:
            logger.error(f"No JSON found in response: {response}")
            raise APIResponseError(f"No JSON found in response: {response}")
            
        json_str = response[json_start:json_end]
        
        # Parse JSON
        result = json.loads(json_str)
        logger.debug(f"Parsed LLM response: {result}")
        return result
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON from response: {response}")
        logger.error(f"JSON decode error: {str(e)}")
        raise APIResponseError(f"Failed to parse JSON from response: {str(e)}")
        
    except Exception as e:
        logger.error(f"Unexpected error in parse_llm_response: {str(e)}")
        raise

def extract_with_llm(text: str, extraction_type: str = "full") -> Dict[str, Any]:
    """
    Extract structured information from medical text using LLM.
    
    Args:
        text: Raw OCR text from medical report
        extraction_type: Type of extraction ("full", "birads", "impression", etc.)
        
    Returns:
        Dictionary with extracted structured data
        
    Raises:
        LLMExtractionError: If extraction fails
    """
    try:
        logger.info(f"Extracting {extraction_type} information using LLM")
        
        # Create appropriate prompt
        prompt = create_extraction_prompt(text, extraction_type)
        
        # Call LLM API
        response = call_groq_api(prompt)
        
        # Parse response
        result = parse_llm_response(response)
        
        logger.info(f"Successfully extracted {extraction_type} information with LLM")
        return result
        
    except Exception as e:
        logger.error(f"Error in extract_with_llm: {str(e)}")
        # Provide a fallback response based on extraction type
        if extraction_type == "birads":
            return {"value": "", "confidence": 0.0}
        elif extraction_type == "sections":
            return {
                "clinical_history": "",
                "findings": "",
                "impression": "",
                "recommendation": ""
            }
        else:
            return {
                "patient_name": "",
                "exam_date": "",
                "exam_type": "",
                "birads_score": "",
                "findings": "",
                "impression": "",
                "recommendation": "",
                "clinical_history": "",
                "provider_info": {
                    "referring_provider": "",
                    "interpreting_provider": "",
                    "facility": ""
                },
                "signed_by": ""
            }

def enhance_extraction_with_llm(extracted_data: Dict[str, Any], raw_text: str) -> Dict[str, Any]:
    """
    Enhance extraction results using LLM when conventional methods fail.
    
    Args:
        extracted_data: Dictionary with extracted fields (may have missing values)
        raw_text: Original raw OCR text
        
    Returns:
        Enhanced data with LLM-extracted fields where needed
    """
    logger.info("Enhancing extraction with LLM")
    
    # Check if we have missing critical fields
    missing_fields = []
    
    # Critical fields that should not be empty
    critical_fields = ["exam_date", "exam_type", "birads_score", "impression"]
    
    for field in critical_fields:
        if field not in extracted_data or not extracted_data[field] or extracted_data[field] == "Not Available":
            missing_fields.append(field)
            
    if not missing_fields:
        logger.info("No critical fields missing, skipping LLM enhancement")
        return extracted_data
        
    logger.info(f"Critical fields missing: {missing_fields}")
    
    # Use targeted extraction for each missing field
    enhanced_data = extracted_data.copy()
    
    # Extract BIRADS if missing
    if "birads_score" in missing_fields:
        try:
            logger.info("Extracting BIRADS score with LLM")
            birads_result = extract_with_llm(raw_text, "birads")
            if birads_result.get("value") and birads_result.get("confidence", 0) > 0.7:
                enhanced_data["birads_score"] = birads_result["value"]
                logger.info(f"LLM extracted BIRADS score: {birads_result['value']} with confidence {birads_result['confidence']}")
        except Exception as e:
            logger.error(f"Failed to extract BIRADS score with LLM: {str(e)}")
    
    # Extract sections if impression or findings are missing
    if "impression" in missing_fields or "findings" in missing_fields:
        try:
            logger.info("Extracting sections with LLM")
            sections_result = extract_with_llm(raw_text, "sections")
            
            # Update missing sections
            for field in ["clinical_history", "findings", "impression", "recommendation"]:
                if field in missing_fields and sections_result.get(field):
                    enhanced_data[field] = sections_result[field]
                    logger.info(f"LLM extracted {field}")
        except Exception as e:
            logger.error(f"Failed to extract sections with LLM: {str(e)}")
    
    # For any remaining critical fields, try full extraction
    if any(field in missing_fields for field in ["exam_date", "exam_type"]):
        try:
            logger.info("Performing full extraction with LLM")
            full_result = extract_with_llm(raw_text, "full")
            
            # Update any remaining missing fields
            for field in missing_fields:
                if field in full_result and full_result[field]:
                    enhanced_data[field] = full_result[field]
                    logger.info(f"LLM extracted {field}")
        except Exception as e:
            logger.error(f"Failed to perform full extraction with LLM: {str(e)}")
    
    logger.info("LLM enhancement completed")
    return enhanced_data 