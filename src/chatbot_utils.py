import pandas as pd
from langdetect import detect
from transformers import pipeline
import re

def analyze_csv(df, diagnosis_pipeline):
    analysis_results = []
    
    # Validate required columns
    required_cols = {'findings', 'birads_score', 'exam_date'}
    if not required_cols.issubset(df.columns):
        raise ValueError("CSV missing required medical columns")
    
    # Analyze each case
    for _, row in df.iterrows():
        case_analysis = {
            'patient_id': row.get('patient_id', 'Unknown'),
            'risk_factors': [],
            'diagnosis_flags': []
        }
        
        # RadBERT analysis
        diagnosis = diagnosis_pipeline(
            f"Findings: {row['findings']}. BIRADS: {row['birads_score']}",
            truncation=True,
            max_length=512
        )
        
        # Process results
        if diagnosis[0]['label'] == 'ABNORMAL':
            case_analysis['diagnosis_flags'].append({
                'type': 'abnormality',
                'confidence': diagnosis[0]['score'],
                'location': 'Unknown'
            })
        
        analysis_results.append(case_analysis)
    
    return analysis_results

def identify_alerts(analysis_results, diagnosis_pipeline):
    alerts = []
    for result in analysis_results:
        # Modified to use passed pipeline instead of creating one
        diagnosis = diagnosis_pipeline(result['observation'])
        # Create alert based on diagnosis
        alert = {
            'observation': result['observation'],
            'diagnosis': diagnosis
        }
        alerts.append(alert)
    return alerts

def generate_chatbot_context(df):
    """Generate medical context summary from dataframe"""
    context = {
        'patient_count': len(df),
        'common_birads': df['birads_score'].mode()[0] if 'birads_score' in df else None,
        'date_range': (df['exam_date'].min(), df['exam_date'].max()) if 'exam_date' in df else None,
        'frequent_findings': df['findings'].value_counts().nlargest(3).to_dict() if 'findings' in df else None,
        'data_quality': {
            'missing_birads': df['birads_score'].isna().sum(),
            'missing_dates': df['exam_date'].isna().sum()
        }
    }
    return context

def generate_medical_response(query, context, chat_history):
    """Generate context-aware medical response with language detection"""
    try:
        lang = detect(query)
    except:
        lang = 'en'  # default to English
    
    # Prepare language-specific prompt
    if lang == 'fr':
        prompt = f"[Contexte Médical]\n{context}\n\n[Historique]\n{chat_history}\n\n[Question] {query}\n[Réponse]"
    else:
        prompt = f"[Medical Context]\n{context}\n\n[History]\n{chat_history}\n\n[Question] {query}\n[Answer]"
    
    # Generate response
    medical_qa = pipeline(
        "text-generation", 
        model="Medical-ChatBot",
        max_length=512,
        temperature=0.3
    )
    
    response = medical_qa(
        prompt,
        max_new_tokens=200,
        do_sample=True,
        clean_up_tokenization_spaces=True
    )[0]['generated_text']
    
    # Extract only the new response
    return response.split("[Réponse]" if lang == 'fr' else "[Answer]")[-1].strip()

def handle_user_feedback(feedback_type, details):
    """Process user feedback about data inconsistencies"""
    feedback_data = {
        'timestamp': pd.Timestamp.now(),
        'feedback_type': feedback_type,
        'details': details
    }
    # Store feedback in database or logging system
    return feedback_data
