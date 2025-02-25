#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 17:25:54 2025

@author: kai_GPT
"""
import os
import cv2
import pytesseract
import pandas as pd
import re
from pdf2image import convert_from_path
from pytesseract import Output
import shutil

def preprocess_image(image_path):
    """Preprocess image for OCR."""
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 3)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

def extract_text_from_image(image):
    """Extract text using Tesseract OCR."""
    custom_config = r'--oem 3 --psm 6'
    return pytesseract.image_to_data(image, config=custom_config, output_type=Output.DICT)

def parse_extracted_text(ocr_data):
    """Parse extracted OCR data into structured format."""
    extracted_text = " ".join([ocr_data['text'][i] for i in range(len(ocr_data['text'])) if int(ocr_data['conf'][i]) > 50])
    # Clean encoding artifacts
    cleaned_text = re.sub(r'[\x80-\xFF]+', '', extracted_text)  # Remove non-ASCII characters
    cleaned_text = cleaned_text.replace('Ã©', 'é').replace('Ã¨', 'è').replace('Ã¢', 'â').replace('Ã', 'a')
    cleaned_text = cleaned_text.replace('â', "'").replace('â', '"').replace('â', '"')
    return cleaned_text

def extract_fields_from_text(text):
    """Extract key fields from the OCR text."""
    fields = {
        'Patient Name': "Unknown",
        'Document Date': "Unknown",
        'Exam Type': "Unknown",
        'Clinical History': "Unknown",
        'BIRADS Right': "Unknown",
        'BIRADS Left': "Unknown",
        'Impressions': "Unknown",
        'Findings': "Unknown",
        'Follow-Up Recommendation': "Unknown",
    }

    # Extract Patient Name
    name_match = re.search(r'Patient[:\s]+([A-Za-z\s]+)', text, re.IGNORECASE)
    if name_match:
        fields['Patient Name'] = name_match.group(1).strip()

    # Extract Document Date
    date_match = re.search(r'Document Date[:\s]+([0-9]{4}-[0-9]{2}-[0-9]{2})', text, re.IGNORECASE)
    if date_match:
        fields['Document Date'] = date_match.group(1).strip()

    # Extract BIRADS Scores
    birads_right_match = re.search(r'BI-RADS RIGHT[:\s]+([0-6])', text, re.IGNORECASE)
    if birads_right_match:
        fields['BIRADS Right'] = birads_right_match.group(1)

    birads_left_match = re.search(r'BI-RADS LEFT[:\s]+([0-6])', text, re.IGNORECASE)
    if birads_left_match:
        fields['BIRADS Left'] = birads_left_match.group(1)

    # Extract Impressions
    impressions_match = re.search(r'IMPRESSION[:\s]+(.*?)\s+(FINDINGS|CONCLUSION)', text, re.IGNORECASE | re.DOTALL)
    if impressions_match:
        fields['Impressions'] = impressions_match.group(1).strip()

    # Extract Findings
    findings_match = re.search(r'FINDINGS[:\s]+(.*?)\s+(RECOMMENDATION|CONCLUSION)', text, re.IGNORECASE | re.DOTALL)
    if findings_match:
        fields['Findings'] = findings_match.group(1).strip()

    return fields

def consolidate_data_by_pdf(extracted_data):
    """Consolidate rows for the same PDF into a single row."""
    consolidated_data = {}

    for row in extracted_data:
        source_pdf = row['Source PDF']
        if source_pdf not in consolidated_data:
            consolidated_data[source_pdf] = row
        else:
            # Merge fields, ensuring no data is lost
            for key, value in row.items():
                if key not in ['Source PDF', 'Page Number', 'Raw OCR Text'] and value != "Unknown":
                    consolidated_data[source_pdf][key] = value

            # Append raw text for all pages
            consolidated_data[source_pdf]['Raw OCR Text'] += f"\nPage {row['Page Number']}: {row['Raw OCR Text']}"

    return list(consolidated_data.values())

def save_to_csv(data, output_file):
    """Save extracted data to a CSV file."""
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)



def process_pdfs_in_folder(folder_path, output_csv, temp_image_dir, save_image_dir):
    """Process all PDF files in a folder and extract data into a CSV."""
    if not os.path.exists(temp_image_dir):
        os.makedirs(temp_image_dir)

    if not os.path.exists(save_image_dir):
        os.makedirs(save_image_dir)

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
                # Save original images
                save_image_path = os.path.join(save_image_dir, f"{os.path.splitext(pdf_file)[0]}_page_{i + 1}.jpg")
                image.save(save_image_path, 'JPEG')

                # Preprocess image
                temp_image_path = os.path.join(temp_image_dir, f"{os.path.splitext(pdf_file)[0]}_page_{i + 1}.jpg")
                image.save(temp_image_path, 'JPEG')
                processed_image = preprocess_image(temp_image_path)

                # Save processed images for inspection
                processed_image_path = os.path.join(save_image_dir, f"{os.path.splitext(pdf_file)[0]}_processed_page_{i + 1}.jpg")
                cv2.imwrite(processed_image_path, processed_image)

                # OCR on processed image
                ocr_data = extract_text_from_image(processed_image)

                # Parse text and extract fields
                extracted_text = parse_extracted_text(ocr_data)
                fields = extract_fields_from_text(extracted_text)
                fields['Source PDF'] = pdf_file  # Add source PDF name for reference
                fields['Page Number'] = i + 1  # Include page number
                fields['Raw OCR Text'] = extracted_text  # Include raw OCR text
                extracted_data.append(fields)

                # Clean up temp image
                os.remove(temp_image_path)

    # Save all extracted data to CSV
    if extracted_data:
        save_to_csv(extracted_data, output_csv)
    else:
        print("No data extracted from PDFs.")

# Example usage
if __name__ == "__main__":
    folder_path = "/home/kai/Desktop/Mammo AI"
    output_csv = "mammogram_results.csv"
    temp_image_dir = "./temp_images"
    save_image_dir = "./saved_images"
    
    process_pdfs_in_folder(folder_path, output_csv, temp_image_dir, save_image_dir)
