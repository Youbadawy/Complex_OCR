import pandas as pd

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
