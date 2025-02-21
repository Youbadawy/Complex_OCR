import cv2
import pytesseract
import re
import pandas as pd
from pytesseract import Output

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    return thresholded

def extract_text_tesseract(image):
    return pytesseract.image_to_string(image, output_type=Output.DICT)

def parse_extracted_text(ocr_result):
    return ocr_result.get('text', '')

def extract_fields_from_text(text, medical_nlp):
    doc = medical_nlp(text)
    # ... rest of original function logic remains same ...
    return pd.DataFrame(fields)
