#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 01:32:16 2025

@author: kai
"""

import os
import pandas as pd
from pdf2image import convert_from_path
import pytesseract
import re
import gc
import torch
from transformers import pipeline

# Paths and setup
folder_path = "/home/kai/Desktop/Mammo AI"  # Folder path
output_csv = "mammogram_results.csv"

# Initialize NLP correction pipeline
nlp_correction = pipeline("text2text-generation", model="google/flan-t5-large", device=0 if torch.cuda.is_available() else -1)

# Helper function: Preprocess image for OCR
def preprocess_image(image_path):
    import cv2
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, processed_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return processed_image

# Extract text using Tesseract OCR
def extract_text_with_tesseract(pdf_path):
    try:
        raw_text = pytesseract.image_to_string(pdf_path, lang="eng", config="--psm 6")
        return raw_text.lower()  # Standardizing to lowercase
    except Exception as e:
        print(f"Error processing {pdf_path} with Tesseract: {e}")
        return ""

# Combine OCR outputs using NLP for corrections
def correct_text_with_nlp(raw_text):
    try:
        corrected_text = nlp_correction(raw_text, max_length=512, truncation=True)[0]['generated_text']
        return corrected_text
    except Exception as e:
        print(f"Error correcting text: {e}")
        return raw_text

# Validate parsed fields for correctness
def validate_and_highlight_field(field_name, field_value):
    if field_name == "dob":
        if not re.match(r"\d{4}-\d{2}-\d{2}", field_value):
            return f"[Potential Error] {field_value}"
    if field_name == "age":
        if not field_value.isdigit():
            return f"[Potential Error] {field_value}"
    return field_value

# Parse fields from text
def parse_fields_from_text(raw_text):
    fields = {
        "patient name": re.search(r"patient[:\s]*([\w\s,-]+)", raw_text),
        "mrn": re.search(r"mrn[:\s]*([\w-]+)", raw_text),
        "dob": re.search(r"dob[:\s]*([\d/-]+)", raw_text),
        "age": re.search(r"age[:\s]*(\d+)", raw_text),
        "sex": re.search(r"sex[:\s]*([\w]+)", raw_text),
        "exam date": re.search(r"exam date[:\s]*([\d/-]+)", raw_text),
        "findings": re.search(r"findings[:\s]*([\w\s,.-]+)", raw_text),
        "impressions": re.search(r"impressions?[:\s]*([\w\s,.-]+)", raw_text),
        "ref phys": re.search(r"referring dr[:\s]*([\w\s]+)", raw_text),
        "status": re.search(r"status[:\s]*([\w\s]+)", raw_text),
        "location": re.search(r"location[:\s]*([\w\s,-]+)", raw_text),
        "hospital": re.search(r"hospital[:\s]*([\w\s,-]+)", raw_text),
        "orders": re.search(r"orders[:\s]*([\w\s,-]+)", raw_text),
        "exam/accession no": re.search(r"accession no[:\s]*([\w\s,-]+)", raw_text),
        "additional results": re.search(r"additional results[:\s]*([\w\s,.-]+)", raw_text),
        "electronically signed by": re.search(r"electronically signed by[:\s]*([\w\s,.-]+)", raw_text),
    }
    return {key: validate_and_highlight_field(key, (match.group(1).strip() if match else "")) for key, match in fields.items()}

# Convert PDF to text and extract data
def process_pdfs_in_folder(folder_path):
    all_data = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".pdf"):
                filepath = os.path.join(root, file)
                try:
                    print(f"Processing PDF: {filepath}")

                    # Extract text using Tesseract OCR
                    raw_text = extract_text_with_tesseract(filepath)

                    # Free GPU memory after OCR
                    gc.collect()
                    torch.cuda.empty_cache()

                    # Correct text using NLP
                    corrected_text = correct_text_with_nlp(raw_text)

                    # Parse structured fields
                    parsed_fields = parse_fields_from_text(corrected_text)

                    all_data.append({
                        "Index": len(all_data) + 1,
                        "Patient Name": parsed_fields.get("patient name", ""),
                        "MRN": parsed_fields.get("mrn", ""),
                        "DOB": parsed_fields.get("dob", ""),
                        "AGE": parsed_fields.get("age", ""),
                        "Sex": parsed_fields.get("sex", ""),
                        "HCN": "",  # Add more parsing if necessary
                        "Patient H-Ph": "",
                        "Account #": "",
                        "orders": parsed_fields.get("orders", ""),
                        "Location": parsed_fields.get("location", ""),
                        "Exam Date": parsed_fields.get("exam date", ""),
                        "Ref Phys": parsed_fields.get("ref phys", ""),
                        "Status": parsed_fields.get("status", ""),
                        "Hospital": parsed_fields.get("hospital", ""),
                        "Exam/Accession No(s)": parsed_fields.get("exam/accession no", ""),
                        "Indication": "",
                        "Findings": parsed_fields.get("findings", ""),
                        "Additional Results": parsed_fields.get("additional results", ""),
                        "Report Status": "",
                        "Clinical": "",
                        "Type of communication": "",
                        "Electronically Signed By": parsed_fields.get("electronically signed by", ""),
                        "Impressions": parsed_fields.get("impressions", ""),
                        "Full Text OCR": corrected_text.strip(),
                    })

                except Exception as e:
                    print(f"Error processing {filepath}: {e}")
                    continue

    df = pd.DataFrame(all_data)
    if not df.empty:
        df.to_csv(output_csv, index=False)
        print(f"Results saved to {output_csv}")
    else:
        print("No data extracted. CSV not created.")

# Main function to run the script
if __name__ == "__main__":
    process_pdfs_in_folder(folder_path)
