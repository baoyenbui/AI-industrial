import json
import re
from datetime import datetime, date

import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os

load_dotenv()

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

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
    f"Procedure:{clean_str(r['ProcedureCode'])} Amount:{clean_str(r['ClaimAmount'])}", axis=1
)

emb   = model.encode(df["text"].tolist(), normalize_embeddings=True)
emb   = np.array(emb, dtype="float32")
index = faiss.IndexFlatIP(emb.shape[1])
index.add(emb)

def safe_float(x):
    try:
        if x is None:
            return None
        x = str(x).replace(",", ".")
        m = re.search(r"\d+(\.\d+)?", x)
        return float(m.group()) if m else None
    except Exception:
        return None


def safe_int(x):
    try:
        if x is None:
            return None
        return int(re.search(r"\d+", str(x)).group())
    except Exception:
        return None


def safe_json(text: str):
    if not text:
        return None
    text = text.strip()
    text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*```$', '', text)
    try:
        return json.loads(text)
    except Exception:
        pass
    try:
        match = re.search(r'(\{.*\})', text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
    except Exception:
        pass
    return None


def clean_query(q):
    if not q:
        return ""
    q = str(q).replace("%3A", ":").replace("%0A", " ")
    return re.sub(r"\s+", " ", q).strip()


def retrieve_similar(query):
    q = clean_query(query)
    if not q:
        return ""
    q_emb = model.encode([q], normalize_embeddings=True)
    q_emb = np.array(q_emb, dtype="float32")
    _, I  = index.search(q_emb, k=3)
    return "\n".join(df.iloc[i]["text"] for i in I[0] if i < len(df))


def get_embedding(text: str):
    if not text:
        return None
    return model.encode([text], normalize_embeddings=True)[0]

def parse_query(query) -> dict:
    if not query:
        return {}

    if isinstance(query, dict):
        data = {}
        for k, v in query.items():
            key = (
                str(k).lower()
                .replace("patient", "")
                .replace("claim", "")
                .replace("_", "")
                .replace(" ", "")
                .strip()
            )
            if v is None or str(v).strip() in ("", "None", "Choose an option", "choose an option"):
                continue
            if isinstance(v, (datetime, date)):
                value = v.strftime("%Y-%m-%d")
            else:
                value = str(v).strip()
            data[key] = value
        return data

    parsed = safe_json(query)
    if isinstance(parsed, dict):
        data = {}
        for k, v in parsed.items():
            key = (
                str(k).lower()
                .replace("patient", "")
                .replace("claim", "")
                .replace("_", "")
                .strip()
            )
            if v is None:
                continue
            if isinstance(v, (datetime, date)):
                value = v.strftime("%Y-%m-%d")
            else:
                value = str(v).strip()
            if value:
                data[key] = value
        return data

    # Input là plain text "Key: Value\n..."
    data = {}
    for line in str(query).split("\n"):
        if ":" in line:
            k, v = line.split(":", 1)
            key = (
                k.strip().lower()
                .replace("patient", "")
                .replace("claim", "")
                .strip()
            )
            value = v.strip()
            if value:
                data[key] = value
    return data


def detect_missing(data: dict) -> list[str]:
    required = ["age", "gender", "income", "employment", "amount"]

    key_mapping = {
        "patientage": "age",       "age": "age",
        "patientgender": "gender", "gender": "gender",
        "patientincome": "income", "income": "income",
        "patientemploymentstatus": "employment",
        "employmentstatus": "employment",
        "employment": "employment",
        "claimamount": "amount",   "amount": "amount",
    }

    normalized = {}
    for k, v in data.items():
        k_lower  = str(k).lower().replace(" ", "").replace("_", "")
        std_key  = key_mapping.get(k_lower, k_lower)
        normalized[std_key] = v

    missing = []
    for k in required:
        v = normalized.get(k)
        if v is None or str(v).strip() == "":
            missing.append(k)
            continue
        v_str = str(v).strip().lower()
        if v_str in ("none", "unknown", "null"):
            missing.append(k)
            continue
        if k in ("age", "amount"):
            try:
                if float(v_str) <= 0:
                    missing.append(k)
            except Exception:
                missing.append(k)
        elif k == "income":
            try:
                if float(v_str) < 0:
                    missing.append(k)
            except Exception:
                missing.append(k)

    return missing