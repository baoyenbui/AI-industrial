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
    "You extract structured line-item data with high precision even from degraded OCR output."
)

VALID_CATEGORIES = {
    "Medicine", "Procedure", "Room", "Lab",
    "Imaging", "Consultation", "Surgery", "Supplies", "Others",
}


def extract_itemized_bill(text: str, diagnosis_code: str = None) -> list[dict]:
    if not text or not text.strip():
        return []

    cleaned = clean_ocr_text(text)
    if len(cleaned) < 20:
        return []

    prompt = f"""Extract every line item from the medical invoice OCR text below.

Diagnosis context: {diagnosis_code or "Unknown"}

HANDLE THESE OCR ISSUES:
- Misaligned columns and broken lines
- Merged cells rendered on separate lines
- Number artifacts: "1O0" → 100, "S500" → $500, "1.500,00" → 1500.00
- Abbreviated descriptions: "Rm & Brd Prv" → "Room and Board Private"
- Missing separators between quantity / unit price / total

CATEGORY RULES (pick exactly one):
- Medicine:     drugs, medications, pharmaceuticals, IV fluids
- Procedure:    treatments, therapy, interventions (non-surgical)
- Room:         room, board, accommodation, nursing care, ward
- Lab:          blood tests, urine tests, cultures, pathology
- Imaging:      X-ray, CT, MRI, ultrasound, PET scan
- Consultation: doctor visits, specialist fees, physician fees
- Surgery:      operating room, surgical procedures, anesthesia
- Supplies:     bandages, gloves, syringes, consumables
- Others:       anything that does not fit above

CONFIDENCE SCORING:
- 0.90+  clearly labeled, no OCR noise
- 0.70-0.89  inferred from context, minor noise
- 0.50-0.69  ambiguous, multiple interpretations
- <0.50  heavily degraded, best guess

FEW-SHOT EXAMPLES:
Input:  "Rm & Brd Prv 01/15/24  1  $850.00  $850.00"
Output: {{"description":"Room and Board Private","quantity":1,"unit_price":850.0,"total":850.0,"category":"Room","code":null,"confidence":0.88}}

Input:  "99213 Office Visit Est. Pt   1   125.00"
Output: {{"description":"Office Visit Established Patient","quantity":1,"unit_price":125.0,"total":125.0,"category":"Consultation","code":"99213","confidence":0.95}}

Input:  "Amoxicil 500mg #30   2   $18.5O"
Output: {{"description":"Amoxicillin 500mg qty 30","quantity":2,"unit_price":18.50,"total":37.0,"category":"Medicine","code":null,"confidence":0.75}}

Input:  "CBC w/ DIFF   1   45.00   45.00"
Output: {{"description":"Complete Blood Count with Differential","quantity":1,"unit_price":45.0,"total":45.0,"category":"Lab","code":null,"confidence":0.90}}

Return ONLY valid JSON, no markdown, no explanation:
{{
  "total_billed": number or null,
  "currency": "USD",
  "ocr_quality": "high" | "medium" | "low",
  "items": [
    {{
      "description": "string",
      "quantity": number or null,
      "unit_price": number or null,
      "total": number,
      "category": "Medicine|Procedure|Room|Lab|Imaging|Consultation|Surgery|Supplies|Others",
      "code": "string or null",
      "confidence": number
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
    quantity   = _safe_float(item.get("quantity"))
    unit_price = _safe_float(item.get("unit_price"))
    total      = _safe_float(item.get("total")) or 0.0

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

    return {
        "description": str(item.get("description", "")).strip(),
        "quantity":    quantity,
        "unit_price":  unit_price,
        "total":       total,
        "category":    category,
        "code":        code,
        "confidence":  round(confidence, 3),
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