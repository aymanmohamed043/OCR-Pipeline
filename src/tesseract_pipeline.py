import pytesseract
import cv2 as cv
import numpy as np
try:
    from src.preprocessing import to_grayscale
except ImportError:
    from preprocessing import to_grayscale


def run_tesseract_ocr(image: np.ndarray, config: str = "") -> str:
    """
    Run Tesseract OCR on an input image and return recognized text.
    This function only performs OCR.
    """
    text = pytesseract.image_to_string(image, config=config)
    return text

def run_tesseract_ocr_pipeline(image: np.ndarray, config: str = "--oem 3 --psm 6") -> str:
    """
    Run the full OCR pipeline: preprocessing, OCR, and postprocessing.
    """
    # gray_image = to_grayscale(image)
    text = run_tesseract_ocr(image, config)
    return text
