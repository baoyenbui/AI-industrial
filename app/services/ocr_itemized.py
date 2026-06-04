import json
import os
import re
import hashlib
from difflib import SequenceMatcher
from groq import Groq
from dotenv import load_dotenv

user_bills_db = {}

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


def normalize_text_value(x) -> str:
    if x is None:
        return ""
    return re.sub(r"\s+", " ", str(x)).strip()


def normalize_text(x) -> str:
    return normalize_text_value(x).lower()


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


def clean_ocr_text(text: str) -> str:
    if not text:
        return ""
    s = str(text)
    s = re.sub(r"[^\w\s\.\:\-\$\,\(\)/\%\#\@]", " ", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = s.strip()
    return s[:8000]


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
    if not code:
        return ""
    base = code[:3]
    full = code[:5] if len(code) >= 5 else None
    if full and full in mapping:
        return mapping[full]
    if base in mapping:
        return mapping[base]
    return f"Disease {code}"


def parse_items_with_regex(text: str) -> list:
    items = []
    pattern = r'(Medical services performed \d+)\s+\$(\d+)\s+(\d{2})\b'
    for match in re.finditer(pattern, text):
        desc  = normalize_text_value(match.group(1)).strip()
        price = int(match.group(2)) + int(match.group(3)) / 100.0
        items.append({
            "description": desc,
            "quantity":    1,
            "unit_price":  round(price, 2),
            "total":       round(price, 2),
            "category":    "Procedure",
            "code":        None,
            "confidence":  0.95,
            "drug_name":   None,
            "disease_name":None,
        })
    if not items:
        pattern2   = r'(\w+[\w\s]*?\d+)\s+\$(\d+)\s+(\d{2})\b'
        skip_words = ['subtotal','tax','total','payment','account','name','bank','item','description','price']
        for match in re.finditer(pattern2, text):
            desc = normalize_text_value(match.group(1)).strip()
            if any(s in desc.lower() for s in skip_words):
                continue
            price = int(match.group(2)) + int(match.group(3)) / 100.0
            if len(desc) >= 3 and 5 <= price <= 500:
                items.append({
                    "description": desc,
                    "quantity":    1,
                    "unit_price":  round(price, 2),
                    "total":       round(price, 2),
                    "category":    "Procedure",
                    "code":        None,
                    "confidence":  0.85,
                    "drug_name":   None,
                    "disease_name":None,
                })
    return items


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
    total      = _safe_float(item.get("total"))
    if quantity and unit_price:
        computed = round(quantity * unit_price, 2)
        if abs(computed - (total or 0)) > 0.10:
            total = computed
    category   = item.get("category", "Others")
    if category not in VALID_CATEGORIES:
        category = "Others"
    confidence = max(0.0, min(1.0, _safe_float(item.get("confidence")) or 0.65))
    return {
        "description":  normalize_text_value(item.get("description")),
        "quantity":     quantity,
        "unit_price":   unit_price,
        "total":        total or 0.0,
        "category":     category,
        "code":         normalize_text_value(item.get("code")),
        "confidence":   round(confidence, 3),
        "drug_name":    normalize_text_value(item.get("drug_name")),
        "disease_name": normalize_text_value(item.get("disease_name")),
    }


def item_fingerprint(item: dict) -> str:
    raw = "|".join([
        normalize_text_value(item.get("description")),
        normalize_text_value(item.get("drug_name")),
        normalize_text_value(item.get("category")),
        str(item.get("quantity")),
        str(item.get("unit_price")),
        str(item.get("total")),
        normalize_text_value(item.get("code")),
    ])
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def dedupe_items(items: list[dict]) -> list[dict]:
    seen, unique = set(), []
    for item in items:
        key = item_fingerprint(item)
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def extract_itemized_bill(text: str, diagnosis_code: str = None) -> list:
    if not text or not text.strip():
        return []
    cleaned = clean_ocr_text(text)
    if len(cleaned) < 20:
        return []

    regex_items = parse_items_with_regex(cleaned)

    if regex_items:
        parsed = {
            "patient_name":  None,
            "total_billed":  None,
            "currency":      "USD",
            "ocr_quality":   "medium",
            "vendor_name":   None,
            "hospital_name": None,
            "bill_date":     None,
            "bill_id":       None,
            "diagnosis_text":None,
            "ocr_text":      cleaned,
            "items":         regex_items,
        }
        m = re.search(r'([A-Z][a-z]+\s+[A-Z][a-z]+)\s+Male', cleaned, re.IGNORECASE)
        if m:
            parsed["patient_name"] = m.group(1)
        m = re.search(
            r'DATE[:\s]+(\d{1,2})\s+(JANUARY|FEBRUARY|MARCH|APRIL|MAY|JUNE|JULY|AUGUST'
            r'|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER|JAN|FEB|MAR|APR|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\s+(\d{4})',
            cleaned, re.IGNORECASE,
        )
        if m:
            parsed["bill_date"] = f"{m.group(1)} {m.group(2).upper()} {m.group(3)}"
        if diagnosis_code:
            parsed["diagnosis_text"] = _map_icd_to_disease_text(diagnosis_code)
        parsed["total_billed"] = round(sum(i.get("total") or 0 for i in regex_items), 2)
        return [parsed]

    disease_context = _map_icd_to_disease_text(diagnosis_code) if diagnosis_code else "Unknown diagnosis"
    prompt = f"""Extract every line item from the medical invoice OCR text below.

Diagnosis context: {disease_context} (ICD code: {diagnosis_code or "Unknown"})

Return ONLY valid JSON:
{{
  "patient_name": "string or null",
  "total_billed": number or null,
  "currency": "USD",
  "ocr_quality": "high" | "medium" | "low",
  "vendor_name": "string or null",
  "hospital_name": "string or null",
  "bill_date": "string or null",
  "bill_id": "string or null",
  "diagnosis_text": "string or null",
  "items": [
    {{
      "description": "string",
      "quantity": number or null,
      "unit_price": number,
      "total": number,
      "category": "Medicine|Procedure|Room|Lab|Imaging|Consultation|Surgery|Supplies|Others",
      "code": "string or null",
      "confidence": number,
      "drug_name": "string or null",
      "disease_name": "string or null"
    }}
  ]
}}

Money format: '$10 00' = 10.00 USD. '$50 00' = 50.00 USD.

OCR TEXT:
{cleaned}"""

    try:
        res = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            temperature=0.0,
            max_tokens=2200,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            response_format={"type": "json_object"},
        )
        raw    = res.choices[0].message.content
        parsed = json.loads(raw)

        items     = parsed.get("items", [])
        validated = [_validate_item(i) for i in items if _is_valid_item(i)]
        parsed["items"] = dedupe_items(validated)

        for k in ["patient_name","vendor_name","hospital_name","bill_date","bill_id","diagnosis_text","currency"]:
            parsed[k] = normalize_text_value(parsed.get(k))
        parsed["currency"]     = parsed["currency"] or "usd"
        parsed["total_billed"] = _safe_float(parsed.get("total_billed"))
        parsed["ocr_text"]     = cleaned
        return [parsed]
    except Exception:
        return []


try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as sk_cosine
    _has_tfidf = True
except Exception:
    _has_tfidf = False


def text_similarity(a: str, b: str) -> float:
    a_norm = normalize_text(a)
    b_norm = normalize_text(b)
    if not a_norm or not b_norm:
        return 0.0
    if _has_tfidf:
        try:
            vec = TfidfVectorizer().fit_transform([a_norm, b_norm])
            return float(sk_cosine(vec[0:1], vec[1:2])[0, 0])
        except Exception:
            pass
    return SequenceMatcher(None, a_norm, b_norm).ratio()


def items_signature(bill: dict) -> dict:
    items  = bill.get("items") or []
    descs  = sorted([normalize_text_value(i.get("description")) for i in items if i.get("description")])
    totals = sorted([str(_safe_float(i.get("total")) or "") for i in items if i.get("total") is not None])
    return {
        "count":     len(items),
        "text":      normalize_text(" | ".join(descs + totals)),
        "set":       set(normalize_text(d) for d in descs if d),
        "total_set": set(totals),
    }


def items_similarity(new: dict, old: dict) -> float:
    n_sig = items_signature(new)
    o_sig = items_signature(old)
    s_n, s_o = n_sig["set"], o_sig["set"]
    jacc = 0.0
    if s_n or s_o:
        inter = len(s_n & s_o)
        union = len(s_n | s_o)
        jacc  = inter / union if union else 0.0
    txt_sim    = text_similarity(n_sig["text"], o_sig["text"])
    t_n, t_o   = n_sig["total_set"], o_sig["total_set"]
    total_jacc = 0.0
    if t_n or t_o:
        t_inter    = len(t_n & t_o)
        t_union    = len(t_n | t_o)
        total_jacc = t_inter / t_union if t_union else 0.0
    return max(jacc, txt_sim * 0.95, total_jacc * 0.9)


def core_signature(bill: dict) -> dict:
    ocr = bill.get("ocr_text") or " ".join(i.get("description", "") for i in bill.get("items", []))
    return {
        "user":      normalize_text(bill.get("user_id")),
        "patient":   normalize_text(bill.get("patient_name")),
        "vendor":    normalize_text(bill.get("vendor_name") or bill.get("hospital_name")),
        "date":      normalize_text(bill.get("bill_date") or ""),
        "amount":    _safe_float(bill.get("total_billed")),
        "currency":  normalize_text(bill.get("currency")),
        "diagnosis": normalize_text(bill.get("diagnosis_text")),
        "text":      normalize_text(ocr),
        "items":     items_signature(bill)["text"],
    }


def duplicate_reason(new_bill: dict, old_bill: dict) -> str | None:
    n = core_signature(new_bill)
    o = core_signature(old_bill)
    if normalize_text(new_bill.get("bill_id")) and normalize_text(new_bill.get("bill_id")) == normalize_text(old_bill.get("bill_id")) and n["vendor"] == o["vendor"]:
        return "same_bill_id_vendor"
    same_pa = n["patient"] and o["patient"] and n["patient"] == o["patient"] and n["amount"] is not None and o["amount"] is not None and abs((n["amount"] or 0.0) - (o["amount"] or 0.0)) < 0.01
    if same_pa and items_similarity(new_bill, old_bill) >= 0.70:
        return "same_patient_amount_high_items_similarity"
    if same_pa and (text_similarity(n["text"], o["text"]) >= 0.75):
        return "same_patient_amount_high_similarity"
    return None


def is_duplicate_bill(new_bill: dict, existing_bills: list[dict]) -> bool:
    for old in existing_bills:
        if duplicate_reason(new_bill, old):
            return True
    return False


def process_ocr_claim(text: str, diagnosis_code: str = None,
                      existing_bills: list[dict] | None = None,
                      user_id: str = None) -> dict:
    existing_bills = existing_bills or []
    bills = extract_itemized_bill(text, diagnosis_code)
    if not bills:
        return {"status": "empty", "bill": None, "duplicate": False, "reason": None}
    bill = bills[0]
    if not bill.get("ocr_text"):
        bill["ocr_text"] = clean_ocr_text(text)
    if user_id:
        bill["user_id"] = user_id
    
    duplicate = is_duplicate_bill(bill, existing_bills)
    reason = None
    if duplicate:
        for old in existing_bills:
            reason = duplicate_reason(bill, old)
            if reason:
                break
        return {"status": "duplicate", "bill": bill, "duplicate": True, "reason": reason}
    
    if user_id:
        if user_id not in user_bills_db:
            user_bills_db[user_id] = []
        user_bills_db[user_id].append(bill)
    
    return {"status": "ok", "bill": bill, "duplicate": False, "reason": None}

