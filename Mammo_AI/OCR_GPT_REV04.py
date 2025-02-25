#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced OCR Script for Mammogram Reports
"""

import os
import cv2
import pytesseract
import pandas as pd
import re
from pdf2image import convert_from_path
from pytesseract import Output
from transformers import pipeline


def preprocess_image(image_path):
    """Preprocess image for OCR."""
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def extract_text_tesseract(image):
    """Extract text using Tesseract OCR."""
    custom_config = r'--oem 3 --psm 6'
    return pytesseract.image_to_data(image, config=custom_config, output_type=Output.DICT)


def parse_extracted_text(ocr_data):
    """Parse extracted OCR data into structured format."""
    extracted_text = " ".join([ocr_data['text'][i] for i in range(len(ocr_data['text'])) if int(ocr_data['conf'][i]) > 50])
    return re.sub(r'[\x80-\xFF]+', '', extracted_text).lower()


def extract_fields_from_text(text, use_medical_nlp=True):
    """Extract key fields from the OCR text dynamically using regex and medical NLP models."""
    fields = {
        'patient name': "unknown",
        'document date': "unknown",
        'exam type': "unknown",
        'clinical history': "unknown",
        'birads right': "unknown",
        'birads left': "unknown",
        'impressions': "unknown",
        'findings': "unknown",
        'follow-up recommendation': "unknown",
    }

    # NLP model for medical context parsing
    if use_medical_nlp:
        medical_nlp = pipeline(
            "ner",
            model="dmis-lab/biobert-base-cased-v1.1",
            tokenizer="dmis-lab/biobert-base-cased-v1.1",
            device=0  # Force GPU usage for this pipeline
        )

        # Truncate or split the text into smaller chunks
        max_length = 256  # Adjusted to handle smaller GPU capacity
        tokens = text.split()  # Split text into tokens
        chunks = [' '.join(tokens[i:i + max_length]) for i in range(0, len(tokens), max_length)]

        nlp_results = []
        for chunk in chunks:
            try:
                nlp_results.extend(medical_nlp(chunk, truncation=True))
            except Exception as e:
                print(f"Error processing chunk: {e}")

        # Extract entities from medical NLP
        for entity in nlp_results:
            if entity['entity'] == 'DATE':
                fields['document date'] = entity['word']
            elif 'BIRADS' in entity['word'].upper():
                if 'right' in entity['word'].lower():
                    fields['birads right'] = entity['word']
                elif 'left' in entity['word'].lower():
                    fields['birads left'] = entity['word']

    # Regex-based parsing as a fallback or supplement
    # Extract Document Date
    date_match = re.search(r'(document date|date of exam)[:\s]+([0-9]{4}-[0-9]{2}-[0-9]{2})', text, re.IGNORECASE)
    if date_match:
        fields['document date'] = date_match.group(2)

    # Extract Exam Type
    exam_match = re.search(r'exam type[:\s]+([a-z\s]+)', text, re.IGNORECASE)
    if exam_match:
        fields['exam type'] = exam_match.group(1).strip()

    # Extract Impressions
    impressions_match = re.search(r'(impressions|conclusion)[:\s]+(.*?)\s+(findings|recommendation)', text, re.IGNORECASE | re.DOTALL)
    if impressions_match:
        fields['impressions'] = impressions_match.group(2).strip()

    # Extract Findings
    findings_match = re.search(r'(findings)[:\s]+(.*?)\s+(recommendation|impression|follow-up)', text, re.IGNORECASE | re.DOTALL)
    if findings_match:
        fields['findings'] = findings_match.group(2).strip()

    # Extract Follow-Up Recommendation
    followup_match = re.search(r'(follow-up|recommendation)[:\s]+(.*?)\s+(signed|printed|end of document)', text, re.IGNORECASE | re.DOTALL)
    if followup_match:
        fields['follow-up recommendation'] = followup_match.group(2).strip()

    return fields


def save_to_csv(data, output_file):
    """Save extracted data to a CSV file."""
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)


def process_pdfs_in_folder(folder_path, output_csv, temp_image_dir):
    """Process all PDF files in a folder and extract data into a CSV."""
    if not os.path.exists(temp_image_dir):
        os.makedirs(temp_image_dir)

    extracted_data = []

    for pdf_file in os.listdir(folder_path):
        if pdf_file.endswith('.pdf'):
            pdf_path = os.path.join(folder_path, pdf_file)

            try:
                images = convert_from_path(pdf_path, dpi=300)
            except Exception as e:
                print(f"Error processing {pdf_file}: {e}")
                continue

            for i, image in enumerate(images):
                temp_image_path = os.path.join(temp_image_dir, f"{os.path.splitext(pdf_file)[0]}_page_{i}.jpg")
                image.save(temp_image_path, 'JPEG')

                # Preprocess image
                processed_image = preprocess_image(temp_image_path)

                # Extract text using Tesseract OCR
                tesseract_data = extract_text_tesseract(processed_image)
                extracted_text = parse_extracted_text(tesseract_data)

                # Parse fields from the extracted text
                fields = extract_fields_from_text(extracted_text)
                fields['source pdf'] = pdf_file  # Add source PDF name for reference
                fields['page number'] = i + 1  # Include page number
                fields['raw ocr text'] = extracted_text  # Include raw OCR text
                extracted_data.append(fields)

                # Clean up temp image
                os.remove(temp_image_path)

    # Save extracted data to CSV
    if extracted_data:
        save_to_csv(extracted_data, output_csv)
    else:
        print("No data extracted from PDFs.")


# Example usage
if __name__ == "__main__":
    folder_path = "/home/kai/Desktop/Mammo AI"  # Path to the folder containing PDF files
    output_csv = "mammogram_results.csv"  # Name of the output CSV file
    temp_image_dir = "./temp_images"  # Temporary directory for saving preprocessed images

    try:
        print(f"Processing PDFs in folder: {folder_path}")
        process_pdfs_in_folder(folder_path, output_csv, temp_image_dir)
        print(f"Processing complete. Results saved to: {output_csv}")
    except Exception as e:
        print(f"An error occurred during processing: {e}")
    finally:
        # Clean up temporary images directory
        if os.path.exists(temp_image_dir):
            for temp_file in os.listdir(temp_image_dir):
                os.remove(os.path.join(temp_image_dir, temp_file))
            os.rmdir(temp_image_dir)
