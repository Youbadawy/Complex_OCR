"""
Model inference functions for Medical Report Processor.
"""
import re
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Union

# Set up logging
logger = logging.getLogger(__name__)

def standardize_birads(df):
    """
    Standardize BIRADS scores in a dataframe
    
    Args:
        df: DataFrame with a 'BIRADS Score' column
    
    Returns:
        DataFrame with standardized BIRADS scores
    """
    if df is None or len(df) == 0:
        return df
    
    if 'BIRADS Score' not in df.columns:
        return df
    
    # Create a copy to avoid modifying the original
    df_copy = df.copy()
    
    # Process each BIRADS score
    if 'BIRADS Score' in df_copy.columns:
        df_copy['BIRADS Score'] = df_copy['BIRADS Score'].apply(extract_birads_number)
    
    return df_copy

def extract_birads_number(value):
    """
    Extract the numeric BIRADS score from various text formats
    
    Args:
        value: Input value (string, number, etc.)
    
    Returns:
        Standardized BIRADS score as a string ('0'-'6') or None
    """
    if value is None:
        return None
        
    # Convert to string for processing
    value_str = str(value).upper().strip()
    
    # Empty string
    if not value_str:
        return None
    
    # Direct number match
    if value_str in ["0", "1", "2", "3", "4", "5", "6"]:
        return value_str
    
    # Look for patterns like "BIRADS 4", "BI-RADS: 4", "Category 4", etc.
    pattern = r'(?:BI-?RADS|CATEGORY|BIRAD)[^0-6]*([0-6])'
    match = re.search(pattern, value_str, re.IGNORECASE)
    
    if match:
        return match.group(1)
    
    # Look for patterns like "4C", "4B", "4A"
    if value_str.startswith("4") and len(value_str) > 1 and value_str[1].upper() in ["A", "B", "C"]:
        return "4"
    
    # Handle English text descriptions
    descriptions = {
        "NEGATIVE": "1",
        "BENIGN": "2",
        "PROBABLY BENIGN": "3",
        "SUSPICIOUS": "4",
        "HIGHLY SUSPICIOUS": "5",
        "HIGHLY SUGGESTIVE OF MALIGNANCY": "5",
        "KNOWN MALIGNANCY": "6"
    }
    
    for desc, score in descriptions.items():
        if desc in value_str:
            return score
    
    # Return original string if no mapping found
    return value_str

def extract_medical_terms(text_series, language='en'):
    """
    Extract medical terms from a pandas Series of text
    
    Args:
        text_series: Series of text strings
        language: Language of the text ('en' for English, 'fr' for French)
    
    Returns:
        Dict of extracted terms by category
    """
    try:
        import nltk
        from nltk.tokenize import word_tokenize
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer
    except ImportError:
        logger.warning("NLTK not available for medical term extraction")
        return {}
    
    # Check if text_series is empty
    if text_series.empty:
        return {}
    
    # Combine all texts
    all_text = ' '.join(text_series.fillna('').astype(str))
    
    # Detect language if not specified
    if language not in ['en', 'fr']:
        # Simple language detection based on common words
        fr_words = ['et', 'le', 'la', 'les', 'du', 'de', 'des', 'un', 'une']
        fr_count = sum(1 for word in fr_words if f' {word} ' in f' {all_text.lower()} ')
        language = 'fr' if fr_count > 3 else 'en'
    
    # Medical term patterns - language-specific
    if language == 'fr':
        patterns = {
            'measurements': r'\d+\s*(?:mm|cm|ml|cc)',
            'birads': r'(?:BI-?RADS|Catégorie)[^0-6]*([0-6][ABC]?)',  # Capture just the BIRADS number
            'locations': r'(?:supérieur|inférieur|externe|interne|central)[e]?[- ](?:quadrant|région)',
            'orientations': r'(?:position|heures)',
            'procedures': r'(?:mammographie|échographie|irm|biopsie|aspiration|excision)',
            'findings': r'(?:masse|calcification|asymétrie|distorsion|rehaussement|hyperplasie|adénose)'
        }
    else:  # English
        patterns = {
            'measurements': r'\d+\s*(?:mm|cm|ml|cc)',
            'birads': r'(?:BI-?RADS|Category)[^0-6]*([0-6][ABC]?)',  # Capture just the BIRADS number
            'locations': r'(?:upper|lower|outer|inner|central)[- ](?:quadrant|region)',
            'orientations': r'(?:clock|o\'clock) position',
            'procedures': r'(?:mammogram|ultrasound|mri|biopsy|aspiration|excision)',
            'findings': r'(?:mass|calcification|asymmetry|distortion|enhancement|hyperplasia|adenosis)'
        }
    
    # Extract terms
    results = {}
    for category, pattern in patterns.items():
        if category == 'birads':
            # For BIRADS, just extract the numeric score
            matches = re.findall(pattern, all_text, re.IGNORECASE)
            # Standardize the BIRADS scores
            birads_scores = [score.upper() for score in matches]
            results[category] = list(set(birads_scores))
        else:
            matches = re.findall(pattern, all_text, re.IGNORECASE)
            # Normalize case for procedures to avoid duplicates like "MAMMOGRAPH" and "mammograph"
            if category == 'procedures':
                matches = [match.lower() for match in matches]
            results[category] = list(set(matches))
    
    # Tokenize and filter for medical terms
    try:
        # Tokenize
        tokens = word_tokenize(all_text.lower())
        
        # Remove stopwords - language specific
        try:
            if language == 'fr':
                stop_words = set(stopwords.words('french'))
            else:
                stop_words = set(stopwords.words('english'))
        except:
            # Fallback if language pack not available
            stop_words = set(['le', 'la', 'les', 'de', 'du', 'des', 'et', 'à', 'au', 'aux', 'un', 'une', 
                             'the', 'a', 'an', 'of', 'in', 'on', 'at', 'by', 'for', 'with', 'about'])
        
        filtered_tokens = [token for token in tokens if token not in stop_words and len(token) > 3]
        
        # Lemmatize (only for English - French would need a different lemmatizer)
        if language == 'en':
            lemmatizer = WordNetLemmatizer()
            lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
        else:
            # For French, just use the filtered tokens
            lemmatized_tokens = filtered_tokens
        
        # Medical term lists - language specific
        if language == 'fr':
            medical_terms = {
                'tissues': ['tissu', 'sein', 'peau', 'mamelon', 'aisselle', 'ganglion', 'lymphatique', 'glande', 'canal', 'lobe'],
                'characteristics': ['dense', 'graisseux', 'hétérogène', 'homogène', 'dispersé', 'spiculé', 'irrégulier', 'circonscrit', 'indistinct'],
                'signs': ['douleur', 'sensibilité', 'gonflement', 'écoulement', 'rougeur', 'chaleur', 'masse', 'nodule', 'épaississement'],
                'diagnoses': ['cancer', 'carcinome', 'malignité', 'malin', 'bénin', 'fibroadénome', 'papillome', 'hyperplasie', 'adénose', 'kyste'],
                'interventions': ['biopsie', 'excision', 'aspiration', 'aiguille', 'stéréotaxique', 'échoguidée']
            }
        else:  # English
            medical_terms = {
                'tissues': ['tissue', 'breast', 'skin', 'nipple', 'axilla', 'lymph', 'node', 'gland', 'duct', 'lobe'],
                'characteristics': ['dense', 'fatty', 'heterogeneous', 'homogeneous', 'scattered', 'spiculated', 'irregular', 'circumscribed', 'indistinct'],
                'signs': ['pain', 'tenderness', 'swelling', 'discharge', 'redness', 'warmth', 'lump', 'nodule', 'thickening'],
                'diagnoses': ['cancer', 'carcinoma', 'malignancy', 'malignant', 'benign', 'fibroadenoma', 'papilloma', 'hyperplasia', 'adenosis', 'cyst'],
                'interventions': ['biopsy', 'excision', 'aspiration', 'fine-needle', 'core', 'stereotactic', 'ultrasound-guided']
            }
        
        # Match tokens to medical terms
        for category, term_list in medical_terms.items():
            matches = [token for token in lemmatized_tokens if token in term_list]
            if category in results:
                results[category].extend(list(set(matches)))
            else:
                results[category] = list(set(matches))
    
    except Exception as e:
        logger.warning(f"Error in NLP processing: {e}")
    
    # Remove empty categories
    results = {k: v for k, v in results.items() if v}
    
    return results

def get_term_explanations(category):
    """
    Get explanations for medical terms by category
    
    Args:
        category: Category of medical terms
    
    Returns:
        Dict of term explanations
    """
    if category == 'birads':
        # Standard detailed BIRADS definitions with recommendations
        return {
            "0": "Incomplete assessment: Additional imaging evaluation or comparison to prior exams is needed.",
            "1": "Negative: No evidence of malignancy. Routine screening recommended.",
            "2": "Benign finding: No evidence of malignancy. Findings such as cysts, calcified fibroadenomas, or other definitively benign lesions. Routine screening recommended.",
            "3": "Probably benign finding: Low probability of malignancy (≤2%). Short-interval follow-up (6 months) or continued surveillance is recommended.",
            "4": "Suspicious abnormality: Tissue diagnosis should be considered. Moderate probability of malignancy (>2% to <95%).",
            "4A": "Low suspicion for malignancy (>2% to ≤10%): Tissue diagnosis required but with low likelihood of malignancy.",
            "4B": "Moderate suspicion for malignancy (>10% to ≤50%): Tissue diagnosis required with intermediate likelihood of malignancy.",
            "4C": "High suspicion for malignancy (>50% to <95%): Tissue diagnosis required with moderate likelihood of malignancy.",
            "5": "Highly suggestive of malignancy: Very high probability of malignancy (≥95%). Appropriate action such as biopsy should be taken.",
            "6": "Known biopsy-proven malignancy: Used for lesions identified as malignant before definitive therapy."
        }
    elif category == 'procedures':
        return {
            "mammogram": "X-ray examination of the breast used to detect and diagnose breast diseases.",
            "mammography": "X-ray examination of the breast used to detect and diagnose breast diseases.",
            "ultrasound": "Imaging technique that uses high-frequency sound waves to visualize internal structures of the breast.",
            "mri": "Magnetic Resonance Imaging; uses magnetic fields and radio waves to create detailed images of breast tissues.",
            "biopsy": "Removal of a sample of tissue for examination under a microscope to check for cancer cells.",
            "aspiration": "Removal of fluid from a cyst using a needle, often guided by ultrasound.",
            "excision": "Surgical removal of tissue, such as a lump or part of the breast.",
            "fine-needle": "Using a thin needle to extract cells for examination, typically for cytology.",
            "core": "Removal of a small cylinder of tissue using a hollow needle for histological examination.",
            "stereotactic": "Using 3D coordinates to locate the exact site for biopsy, particularly useful for calcifications.",
            "ultrasound-guided": "Using ultrasound imaging to guide a biopsy procedure, useful for masses visible on ultrasound."
        }
    elif category == 'findings':
        return {
            "mass": "A space-occupying lesion seen in two different projections, characterized by shape, margin, and density.",
            "calcification": "Calcium deposits in breast tissue that appear as white spots on a mammogram, can be macro or microcalcifications.",
            "asymmetry": "A density visible in one mammographic projection but not in another, can be simple, focal, global, or developing.",
            "distortion": "Abnormal pulling or twisting of the breast tissue without a definite mass, often a sign of malignancy.",
            "enhancement": "Increased brightness in an area of the breast on an MRI after contrast administration, can indicate vascularity.",
            "hyperplasia": "An overgrowth of cells in the ducts or lobules, can be typical or atypical.",
            "adenosis": "Enlargement of lobules in breast tissue, a benign condition but can mimic cancer."
        }
    elif category == 'locations':
        return {
            "upper outer quadrant": "The upper outer portion of the breast (most common location for breast cancer).",
            "upper inner quadrant": "The upper inner portion of the breast, toward the sternum.",
            "lower outer quadrant": "The lower outer portion of the breast.",
            "lower inner quadrant": "The lower inner portion of the breast, toward the sternum.",
            "central region": "The central part of the breast including the nipple and areola."
        }
    # Add more categories as needed
    return {}

def assess_report_quality(report_text: str, extracted_fields: dict = None) -> dict:
    """
    Assess the quality of a medical report based on various criteria.
    
    Args:
        report_text: The text of the medical report
        extracted_fields: Optional dictionary of extracted fields to aid assessment
        
    Returns:
        Dictionary with quality scores and feedback
    """
    if not report_text:
        return {
            "overall_score": 0,
            "completeness": 0,
            "descriptiveness": 0,
            "structure": 0,
            "critical_elements": 0,
            "issues": ["Empty report"],
            "recommendations": ["Provide report content"],
            "has_birads": False,
            "has_measurements": False,
            "has_recommendations": False
        }
    
    # Initialize quality metrics
    scores = {
        "completeness": 0,
        "descriptiveness": 0,
        "structure": 0,
        "critical_elements": 0
    }
    
    issues = []
    recommendations = []
    
    # Assess completeness (presence of key sections)
    key_sections = ["history", "findings", "impression", "recommendation"]
    present_sections = []
    
    for section in key_sections:
        if re.search(rf"\b{section}\b", report_text, re.IGNORECASE):
            present_sections.append(section)
    
    completeness_score = len(present_sections) / len(key_sections) * 100
    scores["completeness"] = min(round(completeness_score), 100)
    
    # Check for missing key sections
    missing_sections = [s for s in key_sections if s not in present_sections]
    if missing_sections:
        issues.append(f"Missing sections: {', '.join(missing_sections)}")
        recommendations.append(f"Add missing sections: {', '.join(missing_sections)}")
    
    # Assess descriptiveness (word count, specialized terms)
    word_count = len(report_text.split())
    
    if word_count < 50:
        descriptiveness = 20
        issues.append("Very brief report")
        recommendations.append("Provide more detailed descriptions")
    elif word_count < 100:
        descriptiveness = 40
        issues.append("Brief report")
        recommendations.append("Consider adding more details")
    elif word_count < 200:
        descriptiveness = 60
    elif word_count < 300:
        descriptiveness = 80
    else:
        descriptiveness = 100
    
    # Check for specialized terms
    medical_terms = ["mass", "calcification", "density", "nodule", "lesion", 
                    "asymmetry", "distortion", "enhancement", "hyperplasia"]
    found_terms = [term for term in medical_terms if term in report_text.lower()]
    
    if len(found_terms) < 3:
        descriptiveness = max(descriptiveness - 20, 0)
        issues.append("Few specialized medical terms")
        recommendations.append("Use more specific medical terminology")
    
    scores["descriptiveness"] = min(round(descriptiveness), 100)
    
    # Assess structure (organization, formatting)
    has_sections = bool(re.search(r"(?:FINDINGS|IMPRESSION|HISTORY|RECOMMENDATION):", report_text, re.IGNORECASE))
    has_paragraphs = report_text.count('\n\n') > 0
    
    if has_sections and has_paragraphs:
        structure_score = 100
    elif has_sections:
        structure_score = 80
    elif has_paragraphs:
        structure_score = 60
    else:
        structure_score = 30
        issues.append("Poor report structure")
        recommendations.append("Organize report into clearly labeled sections")
    
    scores["structure"] = structure_score
    
    # Assess critical elements
    critical_score = 0
    
    # Check for BIRADS
    has_birads = bool(re.search(r"BI-?RADS\s*[0-6]", report_text, re.IGNORECASE))
    if has_birads:
        critical_score += 50
    else:
        issues.append("Missing BI-RADS score")
        recommendations.append("Include BI-RADS assessment")
    
    # Check for recommendations
    has_recommendations = False
    if extracted_fields and 'recommendation' in extracted_fields:
        recommendation = str(extracted_fields['recommendation']).strip()
        has_recommendations = recommendation and recommendation != "N/A"
    else:
        has_recommendations = "recommend" in report_text.lower() or "follow" in report_text.lower()
    
    if has_recommendations:
        critical_score += 25
    else:
        issues.append("No clear recommendations")
        recommendations.append("Include clear follow-up recommendations")
    
    # Check if recommendation matches BIRADS (if available from extracted fields)
    if extracted_fields and 'birads_score' in extracted_fields and 'recommendation' in extracted_fields:
        birads = str(extracted_fields['birads_score']).strip()
        recommendation = str(extracted_fields['recommendation']).strip()
        
        if birads and birads != "N/A" and recommendation and recommendation != "N/A":
            # For BIRADS 4-5, should recommend biopsy
            if birads[0] in "45" and "biopsy" in recommendation.lower():
                # Already counted in has_recommendations
                pass
            # For BIRADS 3, should recommend follow-up
            elif birads[0] == "3" and "follow" in recommendation.lower():
                # Already counted in has_recommendations
                pass
            # For BIRADS 1-2, should recommend routine screening
            elif birads[0] in "12" and "routine" in recommendation.lower():
                # Already counted in has_recommendations
                pass
            else:
                issues.append("Recommendation may not match BI-RADS category")
                recommendations.append("Ensure recommendations are appropriate for the BI-RADS category")
    
    # Check for measurements
    has_measurements = bool(re.search(r"\d+\s*(?:mm|cm)", report_text, re.IGNORECASE))
    if has_measurements:
        critical_score += 25
    else:
        issues.append("No measurements provided")
        recommendations.append("Include measurements for any lesions")
    
    scores["critical_elements"] = critical_score
    
    # Calculate overall score (weighted)
    weights = {
        "completeness": 0.25,
        "descriptiveness": 0.25,
        "structure": 0.2,
        "critical_elements": 0.3
    }
    
    overall_score = sum(scores[metric] * weights[metric] for metric in weights)
    
    # Add overall feedback
    if overall_score >= 90:
        quality_assessment = "Excellent report quality"
    elif overall_score >= 75:
        quality_assessment = "Good report quality"
    elif overall_score >= 60:
        quality_assessment = "Adequate report quality"
    elif overall_score >= 40:
        quality_assessment = "Fair report quality"
    else:
        quality_assessment = "Poor report quality"
    
    # Return the quality assessment
    return {
        "overall_score": round(overall_score),
        "completeness": scores["completeness"],
        "descriptiveness": scores["descriptiveness"],
        "structure": scores["structure"],
        "critical_elements": scores["critical_elements"],
        "assessment": quality_assessment,
        "issues": issues,
        "recommendations": recommendations,
        "has_birads": has_birads,
        "has_measurements": has_measurements,
        "has_recommendations": has_recommendations
    }

def analyze_provider_quality(reports_data: list) -> dict:
    """
    Analyze the quality of reports by provider.
    
    Args:
        reports_data: List of dictionaries with report data including provider information and quality assessments
        
    Returns:
        Dictionary with provider quality analysis
    """
    if not reports_data:
        return {}
    
    # Group reports by provider
    providers = {}
    
    for report in reports_data:
        provider = report.get('interpreting_provider', 'Unknown Provider')
        if provider == '' or provider == 'N/A':
            provider = 'Unknown Provider'
            
        if provider not in providers:
            providers[provider] = {
                'reports': [],
                'report_count': 0,
                'avg_quality': 0,
                'avg_completeness': 0,
                'avg_descriptiveness': 0,
                'avg_structure': 0,
                'avg_critical_elements': 0,
                'birads_usage': {
                    'present': 0,
                    'absent': 0,
                    'percentage': 0
                },
                'measurements_usage': {
                    'present': 0,
                    'absent': 0,
                    'percentage': 0
                },
                'recommendations_usage': {
                    'present': 0,
                    'absent': 0,
                    'percentage': 0
                },
                'specialty_terms_usage': 0,
                'top_issues': {},
                'quality_trend': []
            }
        
        # Add report to provider's list
        providers[provider]['reports'].append(report)
        providers[provider]['report_count'] += 1
        
        # Track quality metrics
        providers[provider]['avg_quality'] += report.get('overall_score', 0)
        providers[provider]['avg_completeness'] += report.get('completeness', 0)
        providers[provider]['avg_descriptiveness'] += report.get('descriptiveness', 0)
        providers[provider]['avg_structure'] += report.get('structure', 0)
        providers[provider]['avg_critical_elements'] += report.get('critical_elements', 0)
        
        # Track critical elements usage
        if report.get('has_birads', False):
            providers[provider]['birads_usage']['present'] += 1
        else:
            providers[provider]['birads_usage']['absent'] += 1
            
        if report.get('has_measurements', False):
            providers[provider]['measurements_usage']['present'] += 1
        else:
            providers[provider]['measurements_usage']['absent'] += 1
            
        if report.get('has_recommendations', False):
            providers[provider]['recommendations_usage']['present'] += 1
        else:
            providers[provider]['recommendations_usage']['absent'] += 1
        
        # Track quality trend over time
        providers[provider]['quality_trend'].append(report.get('overall_score', 0))
        
        # Track issues
        for issue in report.get('issues', []):
            if issue not in providers[provider]['top_issues']:
                providers[provider]['top_issues'][issue] = 0
            providers[provider]['top_issues'][issue] += 1
    
    # Calculate averages and percentages for each provider
    for provider, data in providers.items():
        report_count = data['report_count']
        if report_count > 0:
            data['avg_quality'] = round(data['avg_quality'] / report_count, 1)
            data['avg_completeness'] = round(data['avg_completeness'] / report_count, 1)
            data['avg_descriptiveness'] = round(data['avg_descriptiveness'] / report_count, 1)
            data['avg_structure'] = round(data['avg_structure'] / report_count, 1)
            data['avg_critical_elements'] = round(data['avg_critical_elements'] / report_count, 1)
            
            # Calculate percentages for critical elements
            data['birads_usage']['percentage'] = round(data['birads_usage']['present'] / report_count * 100, 1)
            data['measurements_usage']['percentage'] = round(data['measurements_usage']['present'] / report_count * 100, 1)
            data['recommendations_usage']['percentage'] = round(data['recommendations_usage']['present'] / report_count * 100, 1)
            
            # Sort issues by frequency
            data['top_issues'] = dict(sorted(data['top_issues'].items(), key=lambda x: x[1], reverse=True)[:5])
    
    # Calculate relative performance compared to average
    all_qualities = [data['avg_quality'] for data in providers.values()]
    avg_quality = sum(all_qualities) / len(all_qualities) if all_qualities else 0
    
    # Add relative performance
    for provider, data in providers.items():
        data['relative_performance'] = round(data['avg_quality'] - avg_quality, 1)
        
        # Add grade based on quality score
        if data['avg_quality'] >= 90:
            data['grade'] = 'A'
        elif data['avg_quality'] >= 80:
            data['grade'] = 'B'
        elif data['avg_quality'] >= 70:
            data['grade'] = 'C'
        elif data['avg_quality'] >= 60:
            data['grade'] = 'D'
        else:
            data['grade'] = 'F'
    
    # Sort providers by average quality
    sorted_providers = dict(sorted(providers.items(), key=lambda x: x[1]['avg_quality'], reverse=True))
    
    return sorted_providers 