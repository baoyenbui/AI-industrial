import faiss
import json
import io
import re
import numpy as np
import pandas as pd
from groq import Groq
from sentence_transformers import SentenceTransformer
from PIL import Image
import pytesseract
import os
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("Missing GROQ_API_KEY")

client = Groq(api_key=GROQ_API_KEY)
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

df = pd.read_csv("health_insurance_claims.csv")

cols = [
    "PatientAge","PatientGender","PatientIncome","PatientMaritalStatus",
    "PatientEmploymentStatus","ProviderSpecialty","ClaimAmount",
    "ClaimType","ClaimSubmissionMethod","DiagnosisCode","ProcedureCode","ClaimStatus"
]

df = df[cols].dropna()

df["text"] = df.apply(
    lambda r: f"Age:{r['PatientAge']} Gender:{r['PatientGender']} Income:{r['PatientIncome']} "
              f"Marital:{r['PatientMaritalStatus']} Employment:{r['PatientEmploymentStatus']} "
              f"Specialty:{r['ProviderSpecialty']} Type:{r['ClaimType']} "
              f"Submit:{r['ClaimSubmissionMethod']} Diagnosis:{r['DiagnosisCode']} "
              f"Procedure:{r['ProcedureCode']} Amount:{r['ClaimAmount']}",
    axis=1
)

emb = model.encode(df["text"].tolist())
emb = np.array(emb, dtype="float32")

index = faiss.IndexFlatL2(emb.shape[1])
index.add(emb)

app = FastAPI()

def safe_float(x):
    try:
        if x is None:
            return None
        x = str(x).replace(",", ".")
        m = re.search(r"\d+(\.\d+)?", x)
        return float(m.group()) if m else None
    except:
        return None

def safe_int(x):
    try:
        if x is None:
            return None
        m = re.search(r"\d+", str(x))
        return int(m.group()) if m else None
    except:
        return None

def ocr_image(file_bytes):
    try:
        if not file_bytes:
            return ""
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        text = pytesseract.image_to_string(img)
        return text.strip()
    except:
        return ""

def clean_ocr(text):
    return " ".join(text.split()) if text else ""

def rule_extract(text):
    data = {
        "PatientName": "",
        "PatientAge": None,
        "PatientGender": "",
        "PatientEmploymentStatus": "",
        "PatientMaritalStatus": "",
        "DiagnosisCode": "",
        "ProcedureCode": "",
        "ClaimAmount": None
    }

    if not text:
        return data, 0

    name = re.search(r"([A-Z][a-z]+ [A-Z][a-z]+)\s*\|\s*(male|female)", text, re.I)
    if name:
        data["PatientName"] = name.group(1)

    if re.search(r"\bmale\b", text.lower()) and "female" not in text.lower():
        data["PatientGender"] = "Male"
    elif re.search(r"\bfemale\b", text.lower()):
        data["PatientGender"] = "Female"

    age = re.search(r"(\d{1,3})\s*(?:years?\s*old|yo|age)", text.lower())
    if age:
        data["PatientAge"] = safe_int(age.group(1))

    emp = re.search(r"(employed|unemployed|self[- ]employed)", text.lower())
    if emp:
        data["PatientEmploymentStatus"] = emp.group(1)

    marital = re.search(r"(single|married|divorced|widowed)", text.lower())
    if marital:
        data["PatientMaritalStatus"] = marital.group(1)

    total_matches = re.findall(r"\btotal\b\s*[:$]?\s*\$?\s*([\d.,]+)", text.lower())
    if total_matches:
        value = total_matches[-1].replace(",", ".")
        data["ClaimAmount"] = safe_float(value)

    score = sum(v is not None and str(v).strip() != "" for v in data.values())
    return data, score

def llm_fix(text, partial):
    prompt = f"""
Extract structured data.

TEXT:
{text}

PARTIAL:
{json.dumps(partial)}

Return ONLY JSON:
{{
  "PatientName": "",
  "PatientAge": null,
  "PatientGender": "Male|Female|None",
  "PatientEmploymentStatus": "",
  "PatientMaritalStatus": "",
  "DiagnosisCode": "",
  "ProcedureCode": "",
  "ClaimAmount": null
}}
"""
    try:
        res = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )
        return json.loads(res.choices[0].message.content)
    except:
        return partial

def ocr_pipeline(file: UploadFile):
    try:
        file_bytes = file.file.read()

        if not file_bytes:
            return {"status": "failed", "raw_text": "", "extracted_data": {}}

        text = clean_ocr(ocr_image(file_bytes))

        if not text:
            return {"status": "failed", "raw_text": "", "extracted_data": {}}

        rule_data, score = rule_extract(text)

        if score < 2:
            rule_data = llm_fix(text, rule_data)

        return {
            "status": "ok",
            "raw_text": text,
            "extracted_data": rule_data
        }

    except Exception as e:
        return {
            "status": "failed",
            "raw_text": str(e),
            "extracted_data": {}
        }

def get_claim_approval(query):
    sim = "\n".join(df.iloc[i]["text"] for i in [0, 1, 2])

    prompt = f"""
NEW CLAIM:
{query}

SIMILAR:
{sim}

Return ONLY JSON:
{{
  "decision": "Approved|Denied|Pending",
  "reason": "short",
  "confidence": 0.0
}}
"""

    try:
        res = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )
        return json.loads(res.choices[0].message.content)
    except:
        return {"decision": "Pending", "reason": "parse_error", "confidence": 0.0}