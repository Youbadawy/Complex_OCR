import os
import pandas as pd
from pdf2image import convert_from_path
import pytesseract
import cv2
import numpy as np
from PIL import Image
from paddleocr import PaddleOCR
import easyocr
import torch
import re

# Initialize OCR tools
paddle_ocr = PaddleOCR(use_gpu=torch.cuda.is_available())
easyocr_reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())

# Paths and setup
folder_path = "C:\Users\ybada\OneDrive\Desktop\Chatbots\Medical\Mammo AI"  # Folder path
output_csv = "mammogram_results.csv"
temp_image_dir = "./temp_images"  # Temporary directory for saving images

# Ensure the temporary directory exists
os.makedirs(temp_image_dir, exist_ok=True)

# Helper function: Preprocess image for OCR
def preprocess_image(image_path):
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # Step 1: Adjust DPI
    dpi = 300  # Minimum recommended DPI
    h, w = image.shape[:2]
    dpi_scaling_factor = dpi / 72  # Assuming input images are at 72 DPI
    image = cv2.resize(image, (int(w * dpi_scaling_factor), int(h * dpi_scaling_factor)))

    # Step 2: Deskew text
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    coords = np.column_stack(np.where(gray > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = gray.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    deskewed = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    # Step 3: Fix illumination
    lab = cv2.cvtColor(deskewed, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    illum_fixed = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # Step 4: Binarize and de-noise
    gray = cv2.cvtColor(illum_fixed, cv2.COLOR_BGR2GRAY)
    _, binarized = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    denoised = cv2.fastNlMeansDenoising(binarized, None, 30, 7, 21)

    # Step 5: Ensure correct orientation
    text_orientation = pytesseract.image_to_osd(denoised)
    if "Rotate: 180" in text_orientation:
        denoised = cv2.rotate(denoised, cv2.ROTATE_180)

    return denoised

# Save preprocessed image
def save_preprocessed_image(image, output_path):
    cv2.imwrite(output_path, image)

# Extract text using Tesseract OCR
def extract_text_with_tesseract(image_path):
    try:
        processed_image = preprocess_image(image_path)
        save_preprocessed_image(processed_image, image_path)  # Save processed image for consistency
        raw_text = pytesseract.image_to_string(processed_image, lang="eng", config="--psm 6")
        return raw_text.lower()
    except Exception as e:
        print(f"Error processing image with Tesseract: {e}")
        return ""

# Extract text using EasyOCR
def extract_text_with_easyocr(image_path):
    try:
        image = Image.open(image_path)
        image_np = np.array(image)
        result = easyocr_reader.readtext(image_np, detail=0)
        return " ".join(result).lower()
    except Exception as e:
        print(f"Error processing image with EasyOCR: {e}")
        return ""

# Extract text using PaddleOCR
def extract_text_with_paddleocr(image_path):
    try:
        result = paddle_ocr.ocr(image_path)
        text = " ".join([line[1][0] for line in result[0]])
        return text.lower()
    except Exception as e:
        print(f"Error processing image with PaddleOCR: {e}")
        return ""

# Combine OCR outputs
def combine_ocr_outputs(tesseract_text, easyocr_text, paddleocr_text):
    combined_text = f"{tesseract_text} {easyocr_text} {paddleocr_text}"
    return combined_text

# NLP correction on text
def clean_text_with_nlp(text):
    text = re.sub(r"\bnational defence\b|\bprotected b\b|\bprinted by\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[^a-zA-Z0-9\s.,-]", " ", text)  # Remove unwanted symbols
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    return text

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
        "hospital": re.search(r"hospital[:\s]*([\w\s,-]+)", raw_text),
        "orders": re.search(r"orders[:\s]*([\w\s,-]+)", raw_text),
        "additional results": re.search(r"additional results[:\s]*([\w\s,.-]+)", raw_text),
        "electronically signed by": re.search(r"electronically signed by[:\s]*([\w\s,.-]+)", raw_text),
    }
    return {key: validate_and_highlight_field(key, (match.group(1).strip() if match else "")) for key, match in fields.items()}

# Main function
def main():
    all_data = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".pdf"):
                pdf_path = os.path.join(root, file)
                try:
                    images = convert_from_path(pdf_path)
                    raw_text = ""
                    corrected_text = ""

                    for i, image in enumerate(images):
                        image_path = os.path.join(temp_image_dir, f"page_{i + 1}.png")
                        image.save(image_path, "PNG")
                        processed_image = preprocess_image(image_path)
                        save_preprocessed_image(processed_image, image_path)

                        tesseract_text = extract_text_with_tesseract(image_path)
                        easyocr_text = extract_text_with_easyocr(image_path)
                        paddleocr_text = extract_text_with_paddleocr(image_path)

                        combined_text = combine_ocr_outputs(tesseract_text, easyocr_text, paddleocr_text)
                        raw_text += combined_text + " "
                        corrected_text += clean_text_with_nlp(combined_text) + " "

                    parsed_fields = parse_fields_from_text(corrected_text)
                    all_data.append(parsed_fields)

                except Exception as e:
                    print(f"Error processing {pdf_path}: {e}")

    # Create and save DataFrame
    df = pd.DataFrame(all_data)
    df.to_csv(output_csv, index=False)
    print("Results saved to", output_csv)

if __name__ == "__main__":
    main()
