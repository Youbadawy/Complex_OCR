#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 10:59:07 2025

@author: kai
"""



import os
import pandas as pd
from huggingface_hub import login
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage
import chainlit as cl
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM, pipeline

# Hugging Face authentication
login(token="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")
    

# RadBERT model for medical diagnosis
radbert_model_name = "zzxslp/RadBERT-RoBERTa-4m"
radbert_tokenizer = AutoTokenizer.from_pretrained(radbert_model_name)
radbert_model = AutoModelForSequenceClassification.from_pretrained(radbert_model_name)

# Medical-ChatBot model for conversational purposes
chatbot_model_name = "Mohammed-Altaf/Medical-ChatBot"
chatbot_tokenizer = AutoTokenizer.from_pretrained(chatbot_model_name)
chatbot_model = AutoModelForCausalLM.from_pretrained(chatbot_model_name)

# Initialize pipelines
diagnosis_pipeline = pipeline("text-classification", model=radbert_model, tokenizer=radbert_tokenizer, device=0)  # RadBERT
chatbot_pipeline = pipeline("text-generation", model=chatbot_model, tokenizer=chatbot_tokenizer, device=0)  # Medical-ChatBot

@cl.on_chat_start
async def start():
    """Start the chatbot session."""
    await cl.Message(
        content="👋 Welcome to the Medical Chatbot! Ask questions or upload a CSV of mammograph results for analysis."
    ).send()

@cl.on_message
async def main(message: cl.Message):
    """Handle user messages."""
    user_input = message.content

    if not user_input:
        await cl.Message(content="❓ I didn't understand that. Could you rephrase?").send()
        return

    # Generate a conversational response using the chatbot model
    chatbot_response = chatbot_pipeline(user_input, max_length=150, temperature=0.7, num_return_sequences=1)
    await cl.Message(content=chatbot_response[0]['generated_text']).send()

@cl.file_upload_handler
async def handle_uploaded_file(file: cl.UploadedFile):
    """Handle CSV file uploads for analysis."""
    if file.type != "text/csv":
        await cl.Message(content="⚠️ Please upload a valid CSV file.").send()
        return

    # Save and process the CSV
    csv_path = file.path
    analysis_results = analyze_csv(csv_path)
    await cl.Message(content=f"📊 CSV Analysis Results:\n{analysis_results}").send()

def analyze_csv(file_path):
    """Analyze the CSV file for insights and alerts."""
    df = pd.read_csv(file_path)
    result_summary = {
        "total_records": len(df),
        "missing_values": df.isnull().sum().sum(),
        "critical_alerts": identify_alerts(df),
    }
    return (
        f"Total Records: {result_summary['total_records']}\n"
        f"Missing Values: {result_summary['missing_values']}\n"
        f"Critical Alerts:\n{result_summary['critical_alerts']}"
    )

def identify_alerts(df):
    """Identify critical alerts in the dataset using RadBERT."""
    alerts = []
    if "Findings" in df.columns:
        for _, row in df.iterrows():
            findings_text = row["Findings"]
            if pd.isna(findings_text):
                continue

            # Diagnose using RadBERT
            diagnosis = diagnosis_pipeline(findings_text)
            label = diagnosis[0]["label"]
            score = diagnosis[0]["score"]

            if "Critical" in label:  # Assuming RadBERT outputs a "Critical" label for concerning findings
                alerts.append(f"Critical case in row {row.name}: {row.to_dict()}, Score: {score:.2f}")

    return "\n".join(alerts) if alerts else "No critical cases found."

# Run the chatbot
if __name__ == "__main__":
    cl.run()
