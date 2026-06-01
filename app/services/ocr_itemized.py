import json
import re
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SYSTEM_PROMPT = (
    "You are a medical billing specialist with 20 years of experience "
    "reading hospital invoices, insurance EOBs, and itemized bills from multiple countries. "
    "You extract structured line-item data with high precision even from degraded OCR output. "
    "CRITICAL: You MUST identify DRUG NAMES and map them to the DIAGNOSIS CONTEXT."
)

VALID_CATEGORIES = {
    "Medicine", "Procedure", "Room", "Lab",
    "Imaging", "Consultation", "Surgery", "Supplies", "Others",
}

DRUG_KEYWORDS = [
    "paracetamol", "acetaminophen", "ibuprofen", "diclofenac", "omeprazole", "pantoprazole",
    "amoxicillin", "augmentin", "azithromycin", "cefixime", "ciprofloxacin", "metronidazole",
    "dexamethasone", "prednisolone", "methylprednisolone", "cetirizine", "loratadine",
    "domperidone", "metoclopramide", "omeprazol", "lansoprazole", "rabeprazole",
    "clarithromycin", "levofloxacin", "moxifloxacin", "gentamicin", "vancomycin",
    "oseltamivir", "acyclovir", "valacyclovir", "vitamin c", "vitamin b", "calcium", "zinc",
    "mg", "gram", "g", "ml", "capsule", "tablet", "tab", "cap", "syrup", "injection", "iv",
]

def extract_itemized_bill(text: str, diagnosis_code: str = None) -> list:
    if not text or not text.strip():
        return []

    cleaned = clean_ocr_text(text)
    if len(cleaned) < 20:
        return []

    disease_context = _map_icd_to_disease_text(diagnosis_code) if diagnosis_code else "Unknown diagnosis"

    prompt = f"""Extract every line item from the medical invoice OCR text below.

Diagnosis context: {disease_context} (ICD code: {diagnosis_code or "Unknown"})

CRITICAL RULES FOR DRUG EXTRACTION:
1. If item contains ANY drug keyword (mg, gram, capsule, tablet, or drug names like paracetamol, amoxicillin, omeprazole), mark category="Medicine" and EXTRACT the DRUG NAME explicitly.
2. Drug name extraction: Look for patterns like "Paracetamol 500mg", "Augmentin 625mg", "Omeprazole 20mg". Extract the FULL drug name + dosage.
3. If diagnosis code is provided (e.g., J02 = Acute Pharyngitis), ADD disease context to description: "Acute Pharyngitis - Paracetamol 500mg".
4. Do NOT classify room, board, lab, imaging, consultation as Medicine.

HANDLE THESE OCR ISSUES:
- Misaligned columns and broken lines
- Merged cells rendered on separate lines
- Number artifacts: "1O0" → 100, "S500" → $500, "1.500,00" → 1500.00
- Abbreviated descriptions: "Rm & Brd Prv" → "Room and Board Private", "Amox 500" → "Amoxicillin 500mg"
- Missing separators between quantity / unit price / total

CATEGORY RULES (pick exactly one):
- Medicine: drugs, medications, pharmaceuticals, IV fluids, ANY item with mg/gram/capsule/tablet
- Procedure: treatments, therapy, interventions (non-surgical)
- Room: room, board, accommodation, nursing care, ward
- Lab: blood tests, urine tests, cultures, pathology
- Imaging: X-ray, CT, MRI, ultrasound, PET scan
- Consultation: doctor visits, specialist fees, physician fees
- Surgery: operating room, surgical procedures, anesthesia
- Supplies: bandages, gloves, syringes, consumables
- Others: anything that does not fit above

CONFIDENCE SCORING:
- 0.90+  clearly labeled, no OCR noise
- 0.70-0.89  inferred from context, minor noise
- 0.50-0.69  ambiguous, multiple interpretations
- <0.50  heavily degraded, best guess

FEW-SHOT EXAMPLES:
Input:  "Paracetamol 500mg #30 tabs  1  $5.00  $5.00"  with diagnosis J02
Output: {{"description":"Acute Pharyngitis - Paracetamol 500mg","quantity":1,"unit_price":5.0,"total":5.0,"category":"Medicine","code":null,"confidence":0.95,"drug_name":"Paracetamol 500mg","disease_name":"Acute Pharyngitis"}}

Input:  "Augmentin 625mg #14  2  $25.00  $50.00"  with diagnosis J03
Output: {{"description":"Acute Tonsillitis - Augmentin 625mg","quantity":2,"unit_price":25.0,"total":50.0,"category":"Medicine","code":null,"confidence":0.93,"drug_name":"Augmentin 625mg","disease_name":"Acute Tonsillitis"}}

Input:  "Rm & Brd Prv 01/15/24  1  $850.00  $850.00"
Output: {{"description":"Room and Board Private","quantity":1,"unit_price":850.0,"total":850.0,"category":"Room","code":null,"confidence":0.88,"drug_name":null,"disease_name":null}}

Input:  "CBC w/ DIFF   1   45.00   45.00"
Output: {{"description":"Complete Blood Count with Differential","quantity":1,"unit_price":45.0,"total":45.0,"category":"Lab","code":null,"confidence":0.90,"drug_name":null,"disease_name":null}}

Input:  "Omeprazole 20mg  30 cap  1  $12.00"  with diagnosis K29
Output: {{"description":"Gastritis - Omeprazole 20mg","quantity":1,"unit_price":12.0,"total":12.0,"category":"Medicine","code":null,"confidence":0.92,"drug_name":"Omeprazole 20mg","disease_name":"Gastritis"}}

Return ONLY valid JSON, no markdown, no explanation:
{{
  "total_billed": number or null,
  "currency": "USD",
  "ocr_quality": "high" | "medium" | "low",
  "items": [
    {{
      "description": "string (include disease name if diagnosis_code provided AND this is a Medicine)",
      "quantity": number or null,
      "unit_price": number or null,
      "total": number,
      "category": "Medicine|Procedure|Room|Lab|Imaging|Consultation|Surgery|Supplies|Others",
      "code": "string or null",
      "confidence": number,
      "drug_name": "string or null (extract full drug name + dosage if Medicine)",
      "disease_name": "string or null (only if diagnosis_code provided)"
    }}
  ]
}}

OCR TEXT:
{cleaned}"""

    try:
        res = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            temperature=0.0,
            max_tokens=2000,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            response_format={"type": "json_object"},
        )

        raw    = res.choices[0].message.content
        parsed = json.loads(raw)
        items  = parsed.get("items", [])

        if not items and parsed.get("total_billed"):
            print(
                f"[ocr_itemized] Warning: total_billed={parsed['total_billed']} "
                f"but 0 items extracted. ocr_quality={parsed.get('ocr_quality')}"
            )

        return [_validate_item(i) for i in items if _is_valid_item(i)]

    except json.JSONDecodeError as e:
        print(f"[ocr_itemized] JSON parse error: {e}")
        return []
    except Exception as e:
        print(f"[ocr_itemized] extract_itemized_bill error: {e}")
        return []

def _is_valid_item(item: dict) -> bool:
    if not isinstance(item, dict):
        return False
    desc = str(item.get("description") or "").strip()
    if not desc or len(desc) < 3:
        return False
    total = item.get("total")
    if total is None:
        return False
    try:
        if float(total) < 0:
            return False
    except (TypeError, ValueError):
        return False
    return True

def _validate_item(item: dict) -> dict:
    quantity    = _safe_float(item.get("quantity"))
    unit_price  = _safe_float(item.get("unit_price"))
    total       = _safe_float(item.get("total")) or 0.0

    if quantity and unit_price:
        computed = round(quantity * unit_price, 2)
        if abs(computed - total) > 0.10:
            total = computed

    category = item.get("category", "Others")
    if category not in VALID_CATEGORIES:
        category = "Others"

    confidence = _safe_float(item.get("confidence")) or 0.65
    confidence = max(0.0, min(1.0, confidence))

    code = item.get("code")
    code = str(code).strip() if code and str(code).strip() not in ("None", "null", "") else None

    drug_name = item.get("drug_name")
    drug_name = str(drug_name).strip() if drug_name and str(drug_name).strip() not in ("None", "null", "") else None

    disease_name = item.get("disease_name")
    disease_name = str(disease_name).strip() if disease_name and str(disease_name).strip() not in ("None", "null", "") else None

    return {
        "description": str(item.get("description", "")).strip(),
        "quantity":    quantity,
        "unit_price":  unit_price,
        "total":       total,
        "category":    category,
        "code":        code,
        "confidence":  round(confidence, 3),
        "drug_name":   drug_name,
        "disease_name": disease_name,
    }

def clean_ocr_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"[^\w\s\.\:\-\$\,\(\)/\%\#\@]", " ", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()[:8000]

def _safe_float(x) -> float | None:
    if x is None:
        return None
    try:
        s = str(x).replace(",", "").replace("$", "").strip()
        s = re.sub(r"[^\d\.\-]", "", s)
        if not s:
            return None
        return round(float(s), 2)
    except (ValueError, TypeError):
        return None

def _map_icd_to_disease_text(code: str) -> str:
    mapping = {
        "J00": "Acute nasopharyngitis",
        "J01": "Acute sinusitis",
        "J02": "Acute pharyngitis",
        "J03": "Acute tonsillitis",
        "J04": "Acute laryngitis",
        "J05": "Acute obstructive laryngitis",
        "J06": "Acute upper respiratory infections",
        "J10": "Influenza",
        "J11": "Influenza unspecified",
        "K25": "Gastric ulcer",
        "K26": "Duodenal ulcer",
        "K27": "Peptic ulcer",
        "K29": "Gastritis",
    }
    base = code[:3]
    full = code[:5] if len(code) >= 5 else None
    if full and full in mapping:
        return mapping[full]
    if base in mapping:
        return mapping[base]
    return f"Disease {code}"