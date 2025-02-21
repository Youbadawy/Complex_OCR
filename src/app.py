# Import Streamlit first
import streamlit as st
import time
import platform
import os
# import streamlit_authenticator as stauth  # <- Comment this

# Set page config immediately after import
st.set_page_config(
    page_title="Mammo AI Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Then other imports
import pandas as pd
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers.pipelines import pipeline
from pdf2image import convert_from_path, convert_from_bytes
from PIL import Image
import cv2
import pytesseract
import ocr_utils, chatbot_utils
from huggingface_hub import login
import plotly.express as px
from transformers import MarianTokenizer, MarianMTModel
import concurrent.futures
import logging
from functools import partial
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

login(token="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")

# Add this after imports but before OCR processing
if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
else:
    pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'  # Linux/Mac path

# Configure logging
logging.basicConfig(
    filename=os.path.join(os.path.dirname(__file__), 'mammo_ai.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    force=True
)

def process_single_page(image, page_num, uploaded_file):
    try:
        img_array = np.array(image)
        processed_image = ocr_utils.preprocess_image(img_array)
        
        # OCR processing
        ocr_data = ocr_utils.extract_text_tesseract(processed_image)
        extracted_text = ocr_utils.parse_extracted_text(ocr_data)
        data, warnings = ocr_utils.extract_fields_from_text(extracted_text, medical_nlp, img_array)
        
        # Add metadata
        data.update({
            'source pdf': uploaded_file.name,
            'page number': page_num + 1,
            'raw ocr text': extracted_text,
            'warnings': warnings,
            'patient_name': data.get('patient_name'),
            'document_date': data.get('document_date'),
            # Add all other extracted fields...
        })
        
        return data
    
    except Exception as e:
        logging.error(f"Error processing page {page_num+1} of {uploaded_file.name}", exc_info=True)
        return {
            'error': f"Page {page_num+1}: {str(e)}",
            'source pdf': uploaded_file.name,
            'page number': page_num + 1
        }

# Combined model loader with progress
@st.cache_resource(show_spinner=False)
def load_models():
    with st.spinner("Loading medical AI models..."):
        try:
            # Initialize BioBERT components separately
            from transformers import AutoTokenizer, AutoModelForTokenClassification
            biobert_tokenizer = AutoTokenizer.from_pretrained(
                "dmis-lab/biobert-base-cased-v1.1",
                truncation=True,
                max_length=512
            )
            biobert_model = AutoModelForTokenClassification.from_pretrained(
                "dmis-lab/biobert-base-cased-v1.1"
            )

            return {
                "biobert": pipeline(
                    "ner",
                    model=biobert_model,
                    tokenizer=biobert_tokenizer,
                    device=0 if torch.cuda.is_available() else -1,
                    aggregation_strategy="simple"
                ),
                "radbert": pipeline(
                    "text-classification",
                    model="zzxslp/RadBERT-RoBERTa-4m",
                    device=0 if torch.cuda.is_available() else -1
                ),
                "chatbot": pipeline(
                    "text-generation",
                    model="Mohammed-Altaf/Medical-ChatBot",
                    device=0 if torch.cuda.is_available() else -1,
                    max_length=200,
                    temperature=0.75
                ),
                "biobert_fr": pipeline(
                    "ner",
                    model="Dr-BERT/DrBERT-7GB",
                    device=0 if torch.cuda.is_available() else -1,
                    aggregation_strategy="simple"
                ),
                "medical_ner": pipeline(
                    "ner",
                    model="d4data/biomedical-ner-all",
                    device=0 if torch.cuda.is_available() else -1,
                    aggregation_strategy="simple",
                    framework="pt"
                )
            }
        except Exception as e:
            st.error(f"Model loading failed: {str(e)}")
            raise

# Load all models at once
models = load_models()
medical_nlp = models["biobert"]
diagnosis_pipeline = models["radbert"] 
chatbot_pipeline = models["chatbot"]

# Create main tabs
tab1, tab2, tab3 = st.tabs(["OCR Processing", "Data Analysis", "Chatbot"])

with tab1:
    st.header("OCR Processing")
    st.write("Upload your scanned mammogram reports (PDFs) to extract data.")
    
    # PDF Uploader
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type="pdf",
        accept_multiple_files=True
    )
    
    # Add to OCR Processing tab
    psm_mode = st.selectbox(
        "Layout Analysis Mode",
        options=[
            ("Fully Automatic", "3"),
            ("Single Column", "4"), 
            ("Single Text Block", "6")
        ],
        format_func=lambda x: x[0],
        help="Adjust for different document layouts"
    )
    os.environ["TESS_PSM"] = psm_mode[1]
    
    if st.button("Process PDFs"):
        if not uploaded_files:
            st.warning("Please upload at least one PDF file")
        else:
            with st.spinner("Processing PDFs..."):
                extracted_data = []
                error_messages = []
                template_warnings = []
                
                # Add template matching status
                template_status = st.empty()
                
                # Create progress bar
                progress_bar = st.progress(0)
                total_files = len(uploaded_files)
                
                for file_idx, uploaded_file in enumerate(uploaded_files):
                    template_status.info("🔄 Using image templates for low-confidence fields...")
                    
                    try:
                        images = convert_from_bytes(uploaded_file.read(), dpi=300)
                        
                        with ThreadPoolExecutor() as executor:
                            process_page = partial(process_single_page, uploaded_file=uploaded_file)
                            results = list(executor.map(process_page, images, range(len(images)), chunksize=10))
                        
                        for result in results:
                            if 'error' in result:
                                error_messages.append(result['error'])
                            else:
                                extracted_data.append(result)
                                template_warnings.extend(result.get('template_warnings', []))
                    
                    except Exception as e:
                        error_msg = f"Failed to process {uploaded_file.name}: {str(e)}"
                        logging.error(error_msg, exc_info=True)
                        error_messages.append(error_msg)
                    
                    progress = (file_idx + 1) / total_files
                    progress_bar.progress(progress)
                
                template_status.empty()
                
                # Display template warnings
                if template_warnings:
                    unique_warnings = list(set(template_warnings))
                    with st.expander("⚠️ Template Matching Warnings"):
                        st.write("The following issues were detected during template matching:")
                        for warning in unique_warnings:
                            st.write(f"- {warning}")
                        st.info("Please verify extracted fields and consider updating template images")
                
                # Handle processing results
                if error_messages:
                    st.error("Some pages failed to process:")
                    for msg in error_messages[-3:]:  # Show last 3 errors
                        st.write(f"- {msg}")
                    st.info("Check mammo_ai.log for full details")
                
                if extracted_data:
                    try:
                        # Create dataframe from successful results
                        df = pd.DataFrame([item for sublist in extracted_data for item in sublist])
                        st.session_state['df'] = df
                        st.success(f"Processed {len(extracted_data)} pages successfully")
                        
                    except Exception as e:
                        error_msg = f"Data formatting failed: {str(e)}"
                        logging.error(error_msg, exc_info=True)
                        st.error(error_msg)
    
    # Display results
    if 'df' in st.session_state:
        st.subheader("Processed Results")
        
        # Display structured data
        st.subheader("Structured Medical Data")
        structured_data = ocr_utils.convert_to_structured_json(st.session_state['df'].iloc[0])  # Show first result
        st.json(structured_data)
        
        # Display editable table with dynamic columns
        st.subheader("Full Extracted Data")
        available_columns = [
            col for col in [
                'patient_name', 'document_date', 'birads_score',
                'exam_date', 'document_type', 'testing_provider',
                'mammogram_results', 'additional_information'
            ] if col in st.session_state['df'].columns
        ]
        
        if available_columns:
            df_display = st.session_state['df'][available_columns]
            st.dataframe(df_display, use_container_width=True)
        else:
            st.warning("No standard fields found in extracted data")
            st.write("Raw extracted columns:", st.session_state['df'].columns.tolist())
        
        # Enhanced CSV download
        csv = st.session_state['df'].to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download Full Report",
            data=csv,
            file_name="full_mammogram_report.csv",
            mime="text/csv",
            help="Includes all extracted fields and confidence scores"
        )

with tab2:
    st.header("Data Analysis")
    st.write("Analyze the extracted data or upload your own CSV/Excel file.")
    
    # Data input handling
    if 'df' in st.session_state:
        df = st.session_state['df']
        st.success("Using data from OCR Processing tab.")
    else:
        uploaded_file = st.file_uploader(
            "Upload CSV or Excel file", 
            type=["csv", "xlsx"],
            help="Upload your own medical data for analysis"
        )
        df = pd.read_csv(uploaded_file) if uploaded_file and uploaded_file.name.endswith('.csv') else \
             pd.read_excel(uploaded_file) if uploaded_file and uploaded_file.name.endswith('.xlsx') else None

    if df is not None:
        # Basic statistics
        st.subheader("Basic Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            if 'patient name' in df.columns:
                unique_patients = df['patient name'].nunique()
                st.metric("Unique Patients", unique_patients)
            else:
                st.metric("Unique Patients", "N/A")
        with col2:
            total_reports = len(df)
            st.metric("Total Reports", total_reports)
        with col3:
            latest_date = df['exam date'].max() if 'exam date' in df.columns else "N/A"
            st.metric("Latest Exam Date", latest_date)

        # Exam type analysis
        if 'exam type' in df.columns:
            st.subheader("Exam Type Distribution")
            exam_dist = df['exam type'].value_counts().reset_index()
            exam_dist.columns = ['Exam Type', 'Count']
            st.bar_chart(exam_dist.set_index('Exam Type'))

        # BIRADS analysis
        if {'birads_score', 'exam_date'}.issubset(df.columns):
            st.subheader("BIRADS Analysis")
            
            # Interactive BIRADS score filter
            selected_scores = st.multiselect(
                "Select BIRADS scores to filter:",
                options=['0', '1', '2', '3', '4', '5'],
                default=['4', '5'],
                help="Filter cases by BIRADS score"
            )
            
            # Filter dataframe
            filtered_df = df[df['birads_score'].isin(selected_scores)]
            
            if not filtered_df.empty:
                # Display filtered cases
                st.write(f"Showing {len(filtered_df)} cases with BIRADS {', '.join(selected_scores)}")
                st.dataframe(filtered_df[['patient_name', 'birads_score', 'exam_date']], use_container_width=True)
                
                # Correlation heatmap
                st.subheader("Feature Correlations")
                try:
                    numeric_cols = df.select_dtypes(include=['number']).columns
                    if len(numeric_cols) > 1:
                        corr_matrix = df[numeric_cols].corr()
                        fig = px.imshow(
                            corr_matrix,
                            labels=dict(x="Features", y="Features", color="Correlation"),
                            x=numeric_cols,
                            y=numeric_cols,
                            color_continuous_scale='RdBu',
                            zmin=-1,
                            zmax=1
                        )
                        fig.update_layout(title="Correlation Heatmap of Numeric Features")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Not enough numeric columns for correlation analysis")
                except Exception as e:
                    st.warning(f"Could not generate heatmap: {str(e)}")
            else:
                st.info("No cases found with selected BIRADS scores")

        # Findings analysis
        if 'findings' in df.columns:
            st.subheader("Common Findings")
            findings_text = " ".join(df['findings'].dropna())
            if findings_text:
                word_freq = pd.Series(findings_text.lower().split()).value_counts()[:10]
                st.bar_chart(word_freq)
            else:
                st.write("No findings data available")

        # Interactive exam trends over time
        if 'exam_date' in df.columns:
            st.subheader("Exam Trends Over Time")
            try:
                df['exam_date'] = pd.to_datetime(df['exam_date'])
                df['year'] = df['exam_date'].dt.year
                
                # Year selection slider
                years = sorted(df['year'].unique())
                selected_year = st.slider(
                    "Select Year Range",
                    min_value=min(years),
                    max_value=max(years),
                    value=(min(years), max(years))
                )
                
                # Filter data
                filtered = df[(df['year'] >= selected_year[0]) & (df['year'] <= selected_year[1])]
                monthly_counts = filtered.resample('M', on='exam_date').size().reset_index(name='count')
                
                # Create interactive plot
                fig = px.line(
                    monthly_counts,
                    x='exam_date',
                    y='count',
                    title=f"Exam Trends {selected_year[0]}-{selected_year[1]}",
                    labels={'exam_date': 'Date', 'count': 'Number of Exams'},
                    markers=True
                )
                fig.update_layout(
                    hovermode="x unified",
                    xaxis=dict(showgrid=True),
                    yaxis=dict(showgrid=True)
                )
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.warning(f"Could not analyze trends: {str(e)}")

        # Interactive BIRADS Score Distribution
        st.subheader("BIRADS Score Distribution")
        if 'birads_score' in df.columns:
            try:
                birads_counts = df['birads_score'].value_counts().reset_index()
                birads_counts.columns = ['BIRADS Score', 'Count']
                
                fig = px.bar(
                    birads_counts,
                    x='BIRADS Score',
                    y='Count',
                    color='BIRADS Score',
                    title='BIRADS Score Distribution',
                    labels={'Count': 'Number of Cases'},
                    color_discrete_sequence=px.colors.sequential.Plasma
                )
                fig.update_layout(
                    xaxis={'categoryorder':'total descending'},
                    hovermode='closest'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Add summary statistics
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Cases", len(df))
                col2.metric("Most Common Score", birads_counts.iloc[0]['BIRADS Score'])
                col3.metric("Highest Risk Cases", len(df[df['birads_score'].isin(['4', '5'])]))
                
            except Exception as e:
                st.warning(f"Could not analyze BIRADS scores: {str(e)}")
        else:
            st.warning("BIRADS score data not available for visualization")

        # Additional Information Analysis
        st.subheader("Findings Analysis")
        if 'additional_information' in df.columns:
            try:
                # Generate word cloud
                text = ' '.join(df['additional_information'].dropna())
                wordcloud = WordCloud(width=800, height=400).generate(text)
                
                # Display using matplotlib
                fig, ax = plt.subplots()
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis("off")
                st.pyplot(fig)
                
                # Show frequent terms
                st.write("**Common Clinical Terms**")
                from collections import Counter
                words = re.findall(r'\b[A-Za-z]{4,}\b', text)
                counter = Counter([w.lower() for w in words if w.lower() not in STOP_WORDS])
                st.write(pd.DataFrame(counter.most_common(10), columns=['Term', 'Count']))
                
            except Exception as e:
                st.error(f"Text analysis failed: {str(e)}")

    else:
        st.warning("Patient name column missing in data")
        st.info("ℹ️ No data available. Process PDFs or upload a file to begin analysis.")

with tab3:
    st.header("Chatbot")
    st.write("Ask questions or upload a CSV for analysis.")
    
    # CSV analysis upload
    csv_file = st.file_uploader(
        "Upload CSV for analysis",
        type="csv",
        key="chat_csv",
        help="Upload patient data for contextual analysis"
    )
    
    # Initialize chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg['role']):
            st.markdown(msg['content'])
    
    # Handle CSV analysis
    if csv_file:
        with st.spinner("Analyzing patient data..."):
            try:
                df = pd.read_csv(csv_file)
                analysis_results = chatbot_utils.analyze_csv(df, diagnosis_pipeline)
                
                # Format analysis results
                report = "📊 **Patient Data Analysis Report**\n\n"
                report += f"• Total patients: {len(df)}\n"
                report += f"• Key findings: {analysis_results[:3]}\n"  # Show top 3 findings
                report += "\nAsk me anything about this data!"
                
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'content': report
                })
                st.rerun()
                
            except Exception as e:
                st.error(f"CSV processing failed: {str(e)}")
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'content': "❌ Failed to analyze CSV. Please check the file format."
                })
    
    # Handle user input
    user_input = st.chat_input("Type your message here...")
    if user_input:
        context = ""
        if csv_file:
            context = f"Analyzing {len(df)} patient records. "
        elif 'df' in st.session_state:
            context += f"Report contains: {st.session_state['df']['additional_information'].iloc[0][:200]}... "
        
        # Add BIRADS context
        if 'birads_score' in st.session_state.get('df', pd.DataFrame()).columns:
            context += f"Current BIRADS score: {st.session_state['df']['birads_score'].mode()[0]}. "
        
        full_query = f"{context}\n\nUser question: {user_input}"
        
        # Add user message to history
        st.session_state.chat_history.append({'role': 'user', 'content': user_input})
        
        # Generate response with medical context
        response = chatbot_pipeline(
            full_query,
            max_length=300,
            temperature=0.7,
            num_return_sequences=1
        )[0]['generated_text']
        
        st.session_state.chat_history.append({'role': 'assistant', 'content': response})
        st.rerun()

# Add footer with resource links
st.sidebar.markdown("### Resources")
st.sidebar.markdown("- [Clinical Guidelines](https://example.com)")
st.sidebar.markdown("- [Medical Knowledge Base](https://example.com)")
st.sidebar.markdown("- [Emergency Protocols](https://example.com)")

# Add to sidebar section at bottom of file
st.sidebar.title("Mammogram Analysis Dashboard")
st.sidebar.markdown("""
**Clinical Decision Support System**  
AI-powered analysis of mammogram reports with:
- PDF OCR extraction
- Medical NLP processing
- BIRADS classification
- Clinical insights generation
""")

@st.cache_resource
def load_translation_model():
    return pipeline(
        "translation_fr_to_en",
        model="Helsinki-NLP/opus-mt-fr-en",
        device=0 if torch.cuda.is_available() else -1
    )

if __name__ == "__main__":
    st.write("Medical AI Assistant is running...") 
