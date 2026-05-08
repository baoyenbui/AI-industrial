# app/services/ocr_service.py
import io
import re
from PIL import Image
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def ocr_image(file_bytes: bytes) -> str:
  
    try:
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        text = pytesseract.image_to_string(img, lang='eng+vie')
        return clean_text(text)
    except Exception:
        return ""


def clean_text(t: str) -> str:
    if not t:
        return ""
    t = re.sub(r"[^\w\s\.\:\-\$]", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()