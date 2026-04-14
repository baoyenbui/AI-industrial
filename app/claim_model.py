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
from app.config import GROQ_API_KEY

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

client = Groq(api_key=GROQ_API_KEY)
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

df = pd.read_csv(r"C:\Users\admin\Downloads\ai-claim-approval\health_insurance_claims.csv")

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


def ocr_image(file):
    raw = file.file.read()
    if not raw:
        return ""

    try:
        img = Image.open(io.BytesIO(raw))
        img = img.convert("RGB")
        return pytesseract.image_to_string(img)
    except Exception:
        return ""


def clean_ocr(text):
    return " ".join(text.split()).replace("|", " ").strip()


def rule_extract(text):
    data = {
        "PatientName": None,
        "PatientAge": None,
        "PatientGender": None,
        "PatientEmploymentStatus": None,
        "ClaimAmount": None
    }

    name = re.search(r"\[NAME\](.*?)\[/NAME\]", text, re.I)
    if name:
        data["PatientName"] = name.group(1).strip()

    gender = re.search(r"\[GENDER\](.*?)\[/GENDER\]", text, re.I)
    if gender:
        g = gender.group(1).lower()
        if "male" in g:
            data["PatientGender"] = "M"
        elif "female" in g:
            data["PatientGender"] = "F"

    age = re.search(r"\[AGE\](.*?)\[/AGE\]", text, re.I)
    if age:
        num = re.search(r"\d+", age.group(1))
        if num:
            data["PatientAge"] = int(num.group())

    emp = re.search(r"\[EMPLOYMENT\](.*?)\[/EMPLOYMENT\]", text, re.I)
    if emp:
        data["PatientEmploymentStatus"] = emp.group(1).strip()

    total = re.findall(r"\[TOTAL\](.*?)\[/TOTAL\]", text, re.I)
    if total:
        last = total[-1]
        num = re.search(r"[\d]+[.,]?\d*", last.replace(",", "."))
        if num:
            try:
                data["ClaimAmount"] = float(num.group())
            except:
                pass

    score = sum(v is not None for v in data.values())
    return data, score


def llm_fix(text, partial):
    prompt = f"""
You are a strict extraction system.

Only correct missing fields using OCR text.

OCR TEXT:
{text}

PARTIAL:
{json.dumps(partial)}

Return ONLY JSON:
{{
  "PatientName": null,
  "PatientAge": null,
  "PatientGender": "M|F|null",
  "PatientEmploymentStatus": null,
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


def search_similar(query, k=3):
    q = model.encode([query]).astype("float32")
    _, idx = index.search(q, k)
    return "\n\n".join(df.iloc[i]["text"] for i in idx[0])


def get_claim_approval(query):
    sim = search_similar(query)

    prompt = f"""
NEW CLAIM:
{query}

SIMILAR CASES:
{sim}

Return JSON ONLY:
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
            messages=[
                {"role": "system", "content": "strict insurance engine"},
                {"role": "user", "content": prompt}
            ]
        )
        return json.loads(res.choices[0].message.content)
    except:
        return {"error": "invalid_response"}


def process_claim(data):
    return get_claim_approval(str(data))


def ocr_pipeline(file):
    text = ocr_image(file)

    if not text:
        return {"status": "failed", "reason": "no_ocr"}

    text = clean_ocr(text)

    rule_data, score = rule_extract(text)

    if score < 3:
        rule_data = llm_fix(text, rule_data)

    return {
        "raw_text": text,
        "extracted_data": rule_data,
        "status": "ok"
    }