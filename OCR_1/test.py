from paddleocr import PaddleOCR
import cv2
import re
from datetime import datetime

def preprocess_image(image_path):
    # Load image using OpenCV
    img = cv2.imread(image_path)

    # Check if image was loaded successfully
    if img is None:
        print(f"Error: The image at path '{image_path}' could not be loaded. Please check the file path.")
        return None

    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Optionally save the preprocessed image to inspect
    cv2.imwrite('preprocessed.png', thresh)

    return thresh

def extract_text_from_image(image):
    # Initialize PaddleOCR
    ocr = PaddleOCR(use_angle_cls=True, lang='en')

    # Perform OCR on the preprocessed image
    result = ocr.ocr(image)

    # Print the raw OCR output for debugging
    print("OCR Result:", result)

    # Check if result is not None and contains expected data
    if result and isinstance(result, list) and len(result) > 0 and isinstance(result[0], list):
        # Extract text from the OCR results
        extracted_text = " ".join([line[-1][0] for line in result[0]])
        return extracted_text
    else:
        return "No text detected"

def find_best_before_date(text):
    # Regular expressions to match common date formats
    date_patterns = [
        r'\b\d{2}[./-]\d{2}[./-]\d{4}\b',  # DD/MM/YYYY or DD.MM.YYYY or DD-MM-YYYY
        r'\b\d{2}[./-]\d{2}[./-]\d{2}\b',  # DD/MM/YY or DD.MM.YY or DD-MM-YY
        r'\b\d{4}[./-]\d{2}[./-]\d{2}\b',  # YYYY/MM/DD or YYYY.MM.DD or YYYY-MM-DD
        r'\b\d{6}\b'                      # DDMMYY or MMDDYY
    ]

    for pattern in date_patterns:
        match = re.search(pattern, text)
        if match:
            date_str = match.group(0)
            if len(date_str) == 6:  # Handling DDMMYY or MMDDYY
                try:
                    parsed_date = datetime.strptime(date_str, "%d%m%y")
                    return parsed_date.strftime("%d-%m-%Y")
                except ValueError:
                    return "Date format could not be parsed"
            return date_str

    return "No date found"

# Main execution
image_path = 'bestbefore3.png'  # Replace with your actual image path
preprocessed_image = preprocess_image(image_path)

if preprocessed_image is not None:
    extracted_text = extract_text_from_image(preprocessed_image)
    print("Extracted Text:", extracted_text)

    best_before_date = find_best_before_date(extracted_text)
    print("Best Before Date:", best_before_date)
else:
    print("Image preprocessing failed. Please check the image path and try again.")
