import re
import json
import numpy as np
from groq import Groq
import os
from dotenv import load_dotenv

from ..utils import safe_json, safe_int, safe_float, clean_query, model, retrieve_similar
from ..utils import parse_query, detect_missing

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


MEDICAL_ANCHORS = [
    "medical procedure cpt treatment service billing code",
    "hospital visit therapy diagnosis claim procedure",
]
ADDRESS_ANCHORS = [
    "home address city state zip postal code street avenue road",
    "location new york california texas zip code address",
]

anchor_emb_med  = None
anchor_emb_addr = None


def get_anchors():
    global anchor_emb_med, anchor_emb_addr
    if anchor_emb_med is None:
        anchor_emb_med  = model.encode(MEDICAL_ANCHORS, normalize_embeddings=True)
        anchor_emb_addr = model.encode(ADDRESS_ANCHORS,  normalize_embeddings=True)
    return anchor_emb_med, anchor_emb_addr


def classify_context(context: str) -> float:
    if not context.strip():
        return 0.0
    med_anchors, addr_anchors = get_anchors()
    emb = model.encode([context], normalize_embeddings=True)[0]
    return float(np.max(np.dot(med_anchors, emb)) - np.max(np.dot(addr_anchors, emb)))


def proximity_bonus(context: str) -> float:
    return 0.2 if re.search(r'(procedure|cpt|treatment|service)', context) else 0.0


def rule_extract(text: str) -> tuple[dict, int]:
    t        = text.lower()
    original = text

    data = {
        "PatientAge":             None,
        "PatientGender":          None,
        "PatientEmploymentStatus": None,
        "PatientMaritalStatus":   None,
        "PatientIncome":          None,
        "ProviderSpecialty":      None,
        "ClaimType":              None,
        "ClaimSubmissionMethod":  None,
        "DiagnosisCode":          None,
        "ProcedureCode":          None,
        "ClaimAmount":            None,
        "ClaimStatus":            None,
        "PolicyNumber":           None,
        "DateOfService":          None,
        "HospitalName":           None,
        "PreAuthorizationStatus": None,
    }

    # Gender
    if re.search(r'\b(female|woman|lady|she|her)\b', t):
        data["PatientGender"] = "Female"
    elif re.search(r'\b(male|man|gentleman|he|his)\b', t):
        data["PatientGender"] = "Male"

    # Age
    for pat in [
        r'\bage\s*[:\-]?\s*(\d{1,3})\b',
        r'(\d{1,3})\s*(?:years?\s*old|y\.o\.?|yo|age)',
        r'(\d{1,3})\s*years?',
        r'patient\s*age\s*[:\-]?\s*(\d{1,3})',
    ]:
        m = re.search(pat, t)
        if m:
            val = safe_int(m.group(1))
            if val and 0 < val <= 120:
                data["PatientAge"] = val
                break

    # Employment
    for status, pat in {
        "employed":      r'\b(employed|working|full.time|part.time)\b',
        "self-employed": r'self.?employed|freelance|contractor|business owner',
        "unemployed":    r'unemployed|jobless|not working',
        "student":       r'\bstudent\b',
        "retired":       r'\bretired|pensioner',
    }.items():
        if re.search(pat, t):
            data["PatientEmploymentStatus"] = status
            break

    # Marital
    for status, pat in {
        "single":   r'\b(single|unmarried|never married)\b',
        "married":  r'\b(married|spouse)\b',
        "divorced": r'\b(divorced|separated)\b',
        "widowed":  r'\b(widow|widower|widowed)\b',
    }.items():
        if re.search(pat, t):
            data["PatientMaritalStatus"] = status
            break

    # Income
    m = re.search(r'(?:monthly income|income|salary|earnings)\s*[:\-]?\s*\$?\s*([\d,]+(?:\.\d+)?)', t)
    if m:
        data["PatientIncome"] = safe_float(m.group(1))

    # Claim amount
    for pat in [
        r'(?:total|grand total|claim amount|billed amount|total due|total charged|amount)\s*[:\-]?\s*\$?\s*([\d,]+(?:\.\d+)?)',
        r'\$\s*([\d,]+(?:\.\d+)?)\s*(?:total|usd)',
        r'total\s*[:\-]?\s*\$?\s*([\d,]+(?:\.\d+)?)',
    ]:
        amounts = re.findall(pat, t)
        if amounts:
            data["ClaimAmount"] = safe_float(amounts[-1])
            break

    # Diagnosis code
    diag = re.search(r'\b([A-Z][0-9]{2}(?:\.[0-9A-Z]{1,4})?)\b', original)
    if diag:
        data["DiagnosisCode"] = diag.group(1)

    # Procedure code
    proc       = re.search(r"(?:procedure|cpt|treatment|service|code)\s*[:\-#]?\s*(\d{5})\b", t)
    candidates = []

    if proc:
        code = proc.group(1)
        if int(code) >= 10000:
            start   = max(0, proc.start() - 80)
            end     = min(len(t), proc.end() + 80)
            ctx     = t[start:end]
            if (not re.search(r'\b[a-z\s]+ (ny|ca|tx|fl|nj)?\s*' + code + r'\b', ctx) and
                    not re.search(r'\d+\s+[a-z]+\s+(street|st|road|rd|ave|blvd)', ctx)):
                score = classify_context(ctx) + proximity_bonus(ctx)
                if score > 0.2:
                    data["ProcedureCode"] = code
                else:
                    proc = None
        else:
            proc = None

    if not proc:
        for m in re.finditer(r"\b(\d{5})\b", t):
            code     = m.group(1)
            code_int = int(code)
            if code_int < 10000:
                continue
            start = max(0, m.start() - 80)
            end   = min(len(t), m.end() + 80)
            ctx   = t[start:end]
            if re.search(r'\b[a-z\s]+ (ny|ca|tx|fl|nj)?\s*' + code + r'\b', ctx):
                continue
            if re.search(r'\d+\s+[a-z]+\s+(street|st|road|rd|ave|blvd)', ctx):
                continue
            candidates.append((code, classify_context(ctx) + proximity_bonus(ctx)))
        if candidates:
            best_code, best_score = max(candidates, key=lambda x: x[1])
            if best_score > 0.2:
                data["ProcedureCode"] = best_code

    # Policy number
    for pat in [
        r'(?:policy|member|certificate|subscriber)\s*(?:number|no|id|#)[:\-]?\s*([A-Z0-9\-]{5,30})',
        r'\b([A-Z]{2}\d{6,12})\b',
        r'policy\s*[:\-]?\s*([A-Z0-9\-]{6,25})',
    ]:
        m = re.search(pat, original, re.IGNORECASE)
        if m:
            data["PolicyNumber"] = m.group(1).strip().upper()
            break

    # Date of service
    for pat in [
        r'(?:date of service|dos|service date|treatment date|date)[:\-\s]*(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})',
        r'(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})',
        r'(\d{4}[\/\-\.]\d{1,2}[\/\-\.]\d{1,2})',
        r'(\d{1,2})\s*(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s*(\d{2,4})',
        r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s*(\d{1,2})[,\s]*(\d{2,4})',
        r'(\d{1,2})\s*(january|february|march|april|may|june|july|august|september|october|november|december)\s*(\d{4})',
    ]:
        m = re.search(pat, t)
        if m:
            raw_date = m.group(0).strip()
            data["DateOfService"] = raw_date.title() if len(raw_date) > 6 else m.group(1).strip()
            break

    # Hospital name
    for pat in [
        r'(?:hospital|clinic|medical center|health center|centre|lab|imaging)\s*[:\-]?\s*([A-Za-z0-9\s\&\.\-,]+?)(?=\s*(?:date|policy|amount|patient|invoice|$))',
        r'([A-Za-z0-9\s\&\.\-]+?)\s*(?:hospital|clinic|medical center|healthcare)',
        r'([A-Za-z0-9\s\&\.\-]+?)\s*(?:hosp\.|clinic\.|center)',
    ]:
        m = re.search(pat, original, re.IGNORECASE)
        if m:
            name = m.group(1).strip()
            if len(name) >= 4 and not re.search(r'(total|amount|date|patient)', name.lower()):
                data["HospitalName"] = name.title()
                break

    # Pre-auth
    if re.search(r'pre.?auth(?:orization)?[:\-]?\s*(yes|approved|granted|obtained)', t):
        data["PreAuthorizationStatus"] = "Yes"
    elif re.search(r'pre.?auth(?:orization)?[:\-]?\s*(no|not|denied|pending|required)', t):
        data["PreAuthorizationStatus"] = "No"

    # Submission method
    for method in ["online", "paper", "hospital", "fax", "mail", "electronic", "portal"]:
        if method in t:
            data["ClaimSubmissionMethod"] = method
            break

    # Claim status
    for status in ["approved", "denied", "pending", "appealed", "paid", "rejected", "processed"]:
        if status in t:
            data["ClaimStatus"] = status
            break

    # Specialty
    for s in ["cardiology", "orthopedics", "pediatrics", "neurology", "oncology", "radiology",
              "dermatology", "psychiatry", "gynecology", "urology", "ophthalmology",
              "emergency", "internal medicine", "family medicine", "general surgery"]:
        if s.replace(" ", "") in t.replace(" ", ""):
            data["ProviderSpecialty"] = s
            break

    # Claim type
    for ct in ["medical", "dental", "vision", "pharmacy", "mental health", "maternity", "surgery"]:
        if ct in t:
            data["ClaimType"] = ct
            break

    def valid(v):
        return v is not None and str(v).strip().lower() not in ("", "none", "unknown", "null")

    return data, sum(valid(v) for v in data.values())


def llm_fix(text: str, partial: dict) -> dict:
    prompt = f"""
You are an expert medical claims data extractor.

Return ONLY a valid JSON object. No explanation, no markdown, no extra text.

Rules:
- Use the PARTIAL extraction as a base.
- Correct or fill in any missing/incorrect fields from the TEXT.
- For any field that cannot be clearly determined, use an empty string "".
- Do NOT use "Unknown", "N/A", "null", None, or any other placeholder.
- PatientAge and ClaimAmount must be numbers (integer/float) or "".
- All other fields must be strings or "".

TEXT:
{text}

PARTIAL EXTRACTION (reference only):
{json.dumps(partial, indent=2)}

Required output fields:
PatientAge, PatientGender, PatientEmploymentStatus, PatientMaritalStatus,
PatientIncome, ProviderSpecialty, ClaimType, ClaimSubmissionMethod,
DiagnosisCode, ProcedureCode, ClaimAmount, ClaimStatus,
PolicyNumber, DateOfService, HospitalName, PreAuthorizationStatus

Return only the complete JSON object.
"""
    try:
        res = client.chat.completions.create(
            model="llama-3.1-8b-instant",  
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        parsed = safe_json(res.choices[0].message.content)
        if not isinstance(parsed, dict):
            return partial
        for k in partial:
            if parsed.get(k) is None:
                parsed[k] = partial[k]
        return parsed
    except Exception:
        return partial


def extract_claim_data(raw_text: str) -> dict:
    if not raw_text:
        return {}
    rule_data, score = rule_extract(raw_text)
    if score < 5:
        return llm_fix(raw_text, rule_data)
    return rule_data