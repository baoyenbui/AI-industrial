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

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

df = pd.read_csv("health_insurance_claims.csv")

cols = [
    "PatientAge", "PatientGender", "PatientIncome", "PatientMaritalStatus",
    "PatientEmploymentStatus", "ProviderSpecialty", "ClaimAmount",
    "ClaimType", "ClaimSubmissionMethod", "DiagnosisCode", "ProcedureCode", "ClaimStatus"
]

df = df[cols].fillna("None")


def clean_str(x):
    if x is None:
        return "None"
    x = str(x).strip()
    return x if x else "None"


df["text"] = df.apply(lambda r:
    f"Age:{clean_str(r['PatientAge'])} Gender:{clean_str(r['PatientGender'])} Income:{clean_str(r['PatientIncome'])} "
    f"Marital:{clean_str(r['PatientMaritalStatus'])} Employment:{clean_str(r['PatientEmploymentStatus'])} "
    f"Specialty:{clean_str(r['ProviderSpecialty'])} Type:{clean_str(r['ClaimType'])} "
    f"Submit:{clean_str(r['ClaimSubmissionMethod'])} Diagnosis:{clean_str(r['DiagnosisCode'])} "
    f"Procedure:{clean_str(r['ProcedureCode'])} Amount:{clean_str(r['ClaimAmount'])}",
    axis=1
)

emb = model.encode(df["text"].tolist(), normalize_embeddings=True)
emb = np.array(emb, dtype="float32")

index = faiss.IndexFlatIP(emb.shape[1])
index.add(emb)


def clean_query(q):
    if not q:
        return ""
    q = str(q).replace("%3A", ":").replace("%0A", " ")
    return re.sub(r"\s+", " ", q).strip()


def ocr_image(file_bytes):
    try:
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        return pytesseract.image_to_string(img)
    except:
        return ""


def clean_text(t):
    if not t:
        return ""
    return re.sub(r"[^\w\s\.\:\-\$]", " ", t).strip()


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
        return int(re.search(r"\d+", str(x)).group())
    except:
        return None


def safe_json(text):
    if not text:
        return None
    try:
        return json.loads(text)
    except:
        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            return json.loads(text[start:end])
        except:
            return None
        
MEDICAL_ANCHORS = [
    "medical procedure cpt treatment service billing code",
    "hospital visit therapy diagnosis claim procedure",
]

ADDRESS_ANCHORS = [
    "home address city state zip postal code street avenue road",
    "location new york california texas zip code address",
]

anchor_emb_med = model.encode(MEDICAL_ANCHORS, normalize_embeddings=True)
anchor_emb_addr = model.encode(ADDRESS_ANCHORS, normalize_embeddings=True)

context_cache = {}

def classify_context(context):
    if not context.strip():
        return 0
    if context in context_cache:
        emb = context_cache[context]
    else:
        emb = model.encode([context], normalize_embeddings=True)[0]
        context_cache[context] = emb
    med_score = np.max(np.dot(anchor_emb_med, emb))
    addr_score = np.max(np.dot(anchor_emb_addr, emb))
    return med_score - addr_score

def proximity_bonus(context):
    if re.search(r'(procedure|cpt|treatment|service)', context):
        return 0.2
    return 0

def rule_extract(text):
    t = text.lower()

    data = {
        "PatientAge": None,
        "PatientGender": None,
        "PatientEmploymentStatus": None,
        "PatientMaritalStatus": None,
        "PatientIncome": None,
        "ProviderSpecialty": None,
        "ClaimType": None,
        "ClaimSubmissionMethod": None,
        "DiagnosisCode": None,
        "ProcedureCode": None,
        "ClaimAmount": None,
        "ClaimStatus": None,
    }

    if "female" in t:
        data["PatientGender"] = "Female"
    elif "male" in t:
        data["PatientGender"] = "Male"

    age = None

    age = re.search(r"\bage\s*[:\-]?\s*(\d{1,3})\b", t)

    if not age:
        age = re.search(r"\b(\d{1,3})\s*(?:years?\s*old|y\.o\.?)\b", t)

    if not age:
        for m in re.finditer(r"\b(\d{1,3})\b", t):
            val = safe_int(m.group(1))
            if not val or not (0 < val <= 120):
                continue

            start = max(0, m.start() - 30)
            context = t[start:m.start()]

            if re.search(r"(age|patient|yo|y\.o\.|years?)", context):
                age = m
                break

    if age:
        val = safe_int(age.group(1))
        if val and 0 < val <= 120:
            data["PatientAge"] = val

    emp = re.search(r"(self[- ]employed|unemployed|employed|freelance|contractor|student|retired)", t)
    if emp:
        data["PatientEmploymentStatus"] = emp.group(1)

    marital = re.search(r"(single|married|divorced|widowed)", t)
    if marital:
        data["PatientMaritalStatus"] = marital.group(1)

    income = re.search(r"(?:income|salary|monthly income)\s*[:\-]?\s*\$?\s*([\d,]+(?:\.\d+)?)", t)
    if income:
        data["PatientIncome"] = safe_float(income.group(1))

    for pat in [r"(?:total|amount|claim amount|billed|charged)\s*[:\-]?\s*\$?\s*([\d,]+(?:\.\d+)?)", r"\$\s*([\d,]+(?:\.\d+)?)"]:
        amt = re.findall(pat, t)
        if amt:
            data["ClaimAmount"] = safe_float(amt[-1])
            break

    diag = re.search(r"\b([A-Z]\d{2}(?:\.\d{1,4})?)\b", text)
    if diag:
        data["DiagnosisCode"] = diag.group(1)

    proc = re.search(r"(?:procedure|cpt|treatment|service|code)\s*[:\-#]?\s*(\d{5})\b", t)

    candidates = []

    if proc:
        code = proc.group(1)
        if int(code) >= 10000:
            start = max(0, proc.start() - 80)
            end = min(len(t), proc.end() + 80)
            context = t[start:end]

            if not re.search(r'\b[a-z\s]+ (ny|ca|tx|fl|nj)?\s*' + code + r'\b', context) and \
               not re.search(r'\d+\s+[a-z]+\s+(street|st|road|rd|ave|blvd)', context):

                score = classify_context(context) + proximity_bonus(context)

                if score > 0.2:
                    data["ProcedureCode"] = code
                else:
                    proc = None
        else:
            proc = None

    if not proc:
        for m in re.finditer(r"\b(\d{5})\b", t):
            code = m.group(1)
            code_int = int(code)

            if code_int < 10000:
                continue

            start = max(0, m.start() - 80)
            end = min(len(t), m.end() + 80)
            context = t[start:end]

            if re.search(r'\b[a-z\s]+ (ny|ca|tx|fl|nj)?\s*' + code + r'\b', context):
                continue

            if re.search(r'\d+\s+[a-z]+\s+(street|st|road|rd|ave|blvd)', context):
                continue

            score = classify_context(context) + proximity_bonus(context)
            candidates.append((code, score))

        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            best_code, best_score = candidates[0]

            if best_score > 0.2:
                data["ProcedureCode"] = best_code

    for s in ["cardiology","orthopedics","general surgery","pediatrics","neurology",
              "oncology","radiology","dermatology","psychiatry","gynecology",
              "urology","ophthalmology","emergency","internal medicine","family medicine"]:
        if s in t:
            data["ProviderSpecialty"] = s
            break

    for ct in ["medical","dental","vision","pharmacy","mental health","maternity"]:
        if ct in t:
            data["ClaimType"] = ct
            break

    for sm in ["online","paper","hospital","fax","mail","electronic"]:
        if sm in t:
            data["ClaimSubmissionMethod"] = sm
            break

    for st in ["approved","denied","pending","appealed","paid","rejected"]:
        if st in t:
            data["ClaimStatus"] = st
            break

    def valid(v):
        return v is not None and str(v).strip().lower() not in ["", "none"]

    score = sum(valid(v) for v in data.values())
    return data, score

def llm_fix(text, partial):
    prompt = f"""
Return ONLY valid JSON.

TEXT:
{text}

PARTIAL:
{json.dumps(partial)}

FIELDS:
PatientAge, PatientGender, PatientEmploymentStatus, PatientMaritalStatus, DiagnosisCode, ProcedureCode, ClaimAmount
"""

    try:
        res = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )

        parsed = safe_json(res.choices[0].message.content)

        if not isinstance(parsed, dict):
            return partial

        for k in partial:
            if parsed.get(k) is None:
                parsed[k] = partial[k]

        return parsed

    except:
        return partial

def retrieve_similar(query):
    q = clean_query(query)
    if not q:
        return ""

    q_emb = model.encode([q], normalize_embeddings=True)
    q_emb = np.array(q_emb, dtype="float32")

    _, I = index.search(q_emb, k=3)

    return "\n".join(df.iloc[i]["text"] for i in I[0] if i < len(df))

def parse_query(query):
    data = {}
    for line in query.split("\n"):
        if ":" in line:
            k, v = line.split(":", 1)
            data[k.strip().lower()] = v.strip()
    return data

def detect_missing(data):
    required = [
        "age",
        "gender",
        "income",
        "employment",
        "amount"
    ]

    missing = []

    for k in required:
        v = data.get(k)

        if v is None:
            missing.append(k)
            continue

        v = str(v).strip().lower()

        if v in ["", "none", "unknown"]:
            missing.append(k)
            continue

        if k in ["age", "income", "amount"]:
            try:
                if float(v) <= 0:
                    missing.append(k)
            except:
                missing.append(k)

    return missing

def get_decision(query, sim):

    parsed = parse_query(query)

    missing = detect_missing(parsed)

    if missing:
        return {
            "status": "ok",
            "decision": "Pending",
            "reason": "missing_required_fields",
            "confidence": 0.0
        }

    income = parsed.get("income")
    try:
        if income and float(income) <= 0:
            return {
                "status": "ok",
                "decision": "Denied",
                "reason": "income <= 0",
                "confidence": 1.0
            }
    except:
        pass

    prompt = f"""
Return ONLY JSON.

RULES:
- Otherwise Approved

QUERY:
{query}

SIMILAR:
{sim}
"""
    try:
        res = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )

        parsed = safe_json(res.choices[0].message.content)

        if not parsed:
            return {
                "status": "ok",
                "decision": "Pending",
                "reason": "invalid_llm_output",
                "confidence": 0.0
            }

        parsed["status"] = "ok"
        return parsed

    except Exception as e:
        return {
            "status": "system_error",
            "decision": "Pending",
            "reason": str(e),
            "confidence": 0.0
        }

def get_claim_approval(query):
    query = clean_query(query)
    sim = retrieve_similar(query)
    return get_decision(query, sim)

def ocr_pipeline(file):
    try:
        content = file.file.read()
        text = clean_text(ocr_image(content))

        if not text:
            return {
                "status": "failed",
                "error": "no_text"
            }

        rule_data, score = rule_extract(text)

        if score < 3:
            rule_data = llm_fix(text, rule_data)

        return {
            "status": "ok",
            "raw_text": text,
            "extracted_data": rule_data
        }

    except Exception as e:
        return {
            "status": "system_error",
            "error": str(e)
        }