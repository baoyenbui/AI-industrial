import uuid
import json
from groq import Groq
import os
from dotenv import load_dotenv

from ..utils import safe_float, safe_int, safe_json, parse_query, detect_missing
from .extraction_service import extract_claim_data
from .ocr_itemized import extract_itemized_bill
from .rag_service import vector_search, build_rag_context
from .knowledge_service import upsert_knowledge_item
from .fraud_service import predict_fraud
from .shap_service import explain_decision, format_shap_for_prompt

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

HIGH_RISK_CODES    = {"C50", "C61", "C34", "E11", "I21", "I25"}
PRE_AUTH_THRESHOLD = 5000.0


def _resolve_pre_auth(parsed: dict) -> str:
    keys = ["preauthorizationstatus", "preauthstatus", "preauthorization", "preauth", "pre_auth"]
    for k in keys:
        v = parsed.get(k)
        if v is not None:
            s = str(v).strip().lower()
            if s in ("yes", "true", "1", "approved", "granted"):
                return "Yes"
            if s in ("no", "false", "0", "denied", "not required"):
                return "No"
    return "No"


def _get_amount(parsed: dict, raw: dict) -> float:
    """
    Lookup claim amount across all possible key formats:
    - parsed keys after parse_query strips prefixes (underscore removed):
        "amount"
    - raw dict keys from model_dump() snake_case:
        "claim_amount"
    - raw dict keys from frontend PascalCase payload:
        "ClaimAmount"
    - raw dict keys from OCR extraction PascalCase:
        "ClaimAmount"
    """
    candidates = [
        parsed.get("amount"),
        parsed.get("claimamount"),
        raw.get("claim_amount"),
        raw.get("ClaimAmount"),
        raw.get("claimAmount"),
        raw.get("claimamount"),
        raw.get("Claimamount"),
    ]
    for v in candidates:
        if v is None:
            continue
        result = safe_float(v)
        if result is not None and result > 0:
            return result
    return 0.0


def _get_pre_auth_raw(raw: dict) -> str:
    """Lookup pre_auth from raw dict before parse_query normalization."""
    keys = [
        "pre_authorization_status", "PreAuthorizationStatus",
        "preauthorizationstatus", "pre_auth_status",
        "PreAuthStatus", "pre_auth",
    ]
    for k in keys:
        v = raw.get(k)
        if v is not None:
            s = str(v).strip().lower()
            if s in ("yes", "true", "1", "approved", "granted"):
                return "Yes"
            if s in ("no", "false", "0", "denied", "not required"):
                return "No"
    return ""


def _build_data(parsed: dict, claim_amount: float, raw: dict = None) -> dict:
    raw = raw or {}

    pre_auth = _resolve_pre_auth(parsed)
    if not pre_auth or pre_auth == "No":
        raw_pre = _get_pre_auth_raw(raw)
        if raw_pre:
            pre_auth = raw_pre

    return {
        "claim_amount":       claim_amount,
        "age":                safe_int(parsed.get("patientage") or parsed.get("age") or 0),
        "gender":             str(parsed.get("patientgender") or parsed.get("gender") or raw.get("PatientGender") or ""),
        "claim_type":         str(parsed.get("claimtype") or parsed.get("type") or raw.get("claim_type") or raw.get("ClaimType") or "").lower(),
        "diagnosis_code":     str(parsed.get("diagnosiscode") or parsed.get("diagnosis") or raw.get("diagnosis_code") or raw.get("DiagnosisCode") or ""),
        "procedure_code":     str(parsed.get("procedurecode") or parsed.get("procedure") or raw.get("procedure_code") or raw.get("ProcedureCode") or ""),
        "provider_specialty": str(parsed.get("providerspecialty") or parsed.get("specialty") or raw.get("provider_specialty") or raw.get("ProviderSpecialty") or "").lower(),
        "patient_income":     safe_float(parsed.get("patientincome") or parsed.get("income") or raw.get("patient_income") or raw.get("PatientIncome")) or 0.0,
        "patient_employment": str(parsed.get("patientemploymentstatus") or parsed.get("employment") or raw.get("patient_employment_status") or raw.get("PatientEmploymentStatus") or ""),
        "patient_marital":    str(parsed.get("patientmaritalstatus") or parsed.get("marital") or raw.get("patient_marital_status") or raw.get("PatientMaritalStatus") or ""),
        "policy_number":      str(parsed.get("policynumber") or parsed.get("policyno") or raw.get("policy_number") or raw.get("PolicyNumber") or ""),
        "date_of_service":    str(parsed.get("dateofservice") or parsed.get("date") or raw.get("date_of_service") or raw.get("DateOfService") or ""),
        "hospital_name":      str(parsed.get("hospitalname") or raw.get("hospital_name") or raw.get("HospitalName") or ""),
        "pre_auth":           pre_auth,
        "submission_method":  str(parsed.get("claimsubmissionmethod") or parsed.get("submissionmethod") or raw.get("claim_submission_method") or raw.get("ClaimSubmissionMethod") or ""),
    }


def calculate_baseline_reimbursement(claim_amount: float, data: dict) -> float:
    if claim_amount <= 0:
        return 0.0

    deductible  = 200.0
    coinsurance = 0.25
    copay       = 40.0

    if data.get("age", 0) >= 65:
        deductible  = 150.0
        coinsurance = 0.15

    diagnosis = data.get("diagnosis_code", "")
    if any(code in diagnosis for code in HIGH_RISK_CODES):
        deductible  = 280.0
        coinsurance = 0.30

    if data.get("pre_auth", "No") == "No" and claim_amount > PRE_AUTH_THRESHOLD:
        coinsurance = min(coinsurance + 0.15, 0.90)

    after_deduct  = max(0.0, claim_amount - deductible)
    reimbursement = after_deduct * (1 - coinsurance) - copay
    return max(0.0, reimbursement)


def llm_calculate_reimbursement(
    data: dict,
    baseline: float,
    itemized_items: list = None,
    rag_context: str = "",
    fraud_result: dict = None,
    shap_context: str = "",
) -> dict:
    items_block = json.dumps(itemized_items, indent=2) if itemized_items else "No itemized data"
    item_count  = len(itemized_items) if itemized_items else 0

    fraud_block = ""
    if fraud_result:
        flags = ", ".join(fraud_result.get("flags", [])) or "None"
        fraud_block = f"""
FRAUD RISK ASSESSMENT:
- Score: {fraud_result.get('fraud_score', 0):.4f} / 1.00
- Risk Level: {fraud_result.get('label', 'LOW_RISK')}
- Flags: {flags}
- If HIGH_RISK: reduce confidence, flag for manual review, consider Partial Approval.
- If LOW_RISK: proceed normally.
"""

    prompt = f"""You are a strict but fair senior health insurance claims auditor.

CLAIM DETAILS:
- Claim Amount:        ${data.get('claim_amount', 0):,}
- Pre-Authorization:   {data.get('pre_auth', 'No')}
- Policy Number:       {data.get('policy_number') or 'N/A'}
- Patient Age:         {data.get('age', 'N/A')}
- Hospital:            {data.get('hospital_name') or 'N/A'}
- Diagnosis Code:      {data.get('diagnosis_code') or 'N/A'}
- Procedure Code:      {data.get('procedure_code') or 'N/A'}
- Provider Specialty:  {data.get('provider_specialty') or 'N/A'}
- Rule-based Baseline: ${baseline:,.2f}

ITEMIZED BILL ({item_count} items):
{items_block}

KNOWLEDGE BASE CONTEXT:
{rag_context or 'No similar cases found.'}
{fraud_block}
{shap_context}

STRICT RULES — apply in this exact order:
1. A missing Policy Number does NOT affect the decision. Ignore it completely.
2. A missing Diagnosis Code → decision must be "Pending", never "Denied".
3. Pre-Auth = "Yes" → Approve. May flag unreasonable line items.
4. Pre-Auth = "No" AND Claim > ${PRE_AUTH_THRESHOLD:,.0f} → Deny. reimbursement = 0.
5. Pre-Auth = "No" AND Claim <= ${PRE_AUTH_THRESHOLD:,.0f} → Approve at 70% of baseline.
6. Flag any line item priced >30% above knowledge base average.
7. Final reimbursement = 85% of eligible baseline unless rules above override.
8. Explanation must be written for the patient — include the actual dollar amounts calculated.

Return ONLY valid JSON, no markdown:
{{
  "decision": "Approved" | "Partially Approved" | "Denied" | "Pending",
  "reimbursement_amount": number,
  "eligible_amount": number,
  "reason": "concise technical reason",
  "explanation": "clear patient-facing explanation with exact calculation breakdown",
  "flagged_items": [],
  "policy_type": "standard" | "premium" | "high_risk",
  "confidence": 0.00
}}"""

    try:
        res = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            temperature=0.0,
            max_tokens=800,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        result = safe_json(res.choices[0].message.content)
        if isinstance(result, dict) and result.get("decision"):
            return result
    except Exception as e:
        print(f"[claim_service] LLM error: {e}")

    return _fallback_decision(data, baseline)


def _fallback_decision(data: dict, baseline: float) -> dict:
    pre_auth  = data.get("pre_auth", "No")
    amount    = float(data.get("claim_amount", 0))

    if not data.get("diagnosis_code"):
        return {
            "decision":             "Pending",
            "reimbursement_amount": 0.0,
            "eligible_amount":      0.0,
            "reason":               "missing_diagnosis_code",
            "explanation":          "Your claim is pending because a valid diagnosis code (ICD-10) was not provided. Please resubmit with the code from your doctor.",
            "flagged_items":        [],
            "policy_type":          "standard",
            "confidence":           0.60,
        }

    reimburse = round(baseline * 0.85, 2)

    if pre_auth == "Yes":
        return {
            "decision":             "Approved",
            "reimbursement_amount": reimburse,
            "eligible_amount":      reimburse,
            "reason":               "pre_authorization_confirmed",
            "explanation":          f"Your pre-authorization was confirmed.\nBaseline: ${baseline:,.2f} × 85% = ${reimburse:,.2f}",
            "flagged_items":        [],
            "policy_type":          "standard",
            "confidence":           0.82,
        }

    if amount > PRE_AUTH_THRESHOLD:
        return {
            "decision":             "Denied",
            "reimbursement_amount": 0.0,
            "eligible_amount":      0.0,
            "reason":               "no_pre_auth_exceeds_threshold",
            "explanation":          f"Your claim of ${amount:,.2f} was denied because it exceeds ${PRE_AUTH_THRESHOLD:,.0f} and no pre-authorization was obtained before treatment.",
            "flagged_items":        [],
            "policy_type":          "standard",
            "confidence":           0.90,
        }

    reduced = round(baseline * 0.70, 2)
    return {
        "decision":             "Approved",
        "reimbursement_amount": reduced,
        "eligible_amount":      reduced,
        "reason":               "no_pre_auth_under_threshold",
        "explanation":          f"Approved at reduced rate (no pre-authorization).\nBaseline: ${baseline:,.2f} × 70% = ${reduced:,.2f}",
        "flagged_items":        [],
        "policy_type":          "standard",
        "confidence":           0.75,
    }


def get_decision(
    query=None,
    rag_context: str = "",
    itemized_items: list = None,
    fraud_result: dict = None,
    shap_context: str = "",
) -> dict:
    raw    = query if isinstance(query, dict) else {}
    parsed = parse_query(query)

    missing = detect_missing(parsed)
    if missing:
        return {
            "status":               "ok",
            "decision":             "Pending",
            "reason":               "missing_required_fields",
            "missing_fields":       missing,
            "reimbursement_amount": 0.0,
            "baseline_amount":      0.0,
            "explanation":          "Some required fields are missing. Please provide complete claim information.",
            "confidence":           0.65,
        }

    claim_amount = _get_amount(parsed, raw)
    data         = _build_data(parsed, claim_amount, raw)
    baseline     = calculate_baseline_reimbursement(claim_amount, data)

    print(f"[get_decision] amount={claim_amount} baseline={baseline} pre_auth={data['pre_auth']} diag={data['diagnosis_code']}")

    llm_result = llm_calculate_reimbursement(
        data, baseline, itemized_items or [], rag_context, fraud_result, shap_context
    )

    final_amount = float(llm_result.get("reimbursement_amount", baseline))

    if data["pre_auth"] == "No" and claim_amount > PRE_AUTH_THRESHOLD:
        final_amount = 0.0
    elif baseline > 0:
        min_allowed  = baseline * 0.65
        max_allowed  = max(baseline * 1.35, claim_amount)
        final_amount = max(min_allowed, min(max_allowed, final_amount))

    return {
        "status":               "ok",
        "decision":             llm_result.get("decision", "Pending"),
        "reason":               llm_result.get("reason", "processed_by_hybrid_system"),
        "reimbursement_amount": round(final_amount, 2),
        "baseline_amount":      round(baseline, 2),
        "adjustment_percent":   round((final_amount / baseline - 1) * 100, 1) if baseline > 0 else 0,
        "explanation":          llm_result.get("explanation", f"Standard reimbursement: ${baseline:,.2f}"),
        "flagged_items":        llm_result.get("flagged_items", []),
        "policy_type":          llm_result.get("policy_type", "standard"),
        "confidence":           llm_result.get("confidence", 0.75),
        "policy_number":        data["policy_number"],
        "date_of_service":      data["date_of_service"],
        "hospital_name":        data["hospital_name"],
        "pre_auth":             data["pre_auth"],
    }


def process_claim(
    file_bytes: bytes = None,
    form_data: dict = None,
    db_knowledge=None,
    db_claims=None,
) -> dict:
    raw_text = ""
    if file_bytes:
        from .ocr_service import ocr_image, clean_text
        raw_text = clean_text(ocr_image(file_bytes))

    extracted_data = extract_claim_data(raw_text) if raw_text else (form_data or {})

    parsed       = parse_query(extracted_data)
    claim_amount = _get_amount(parsed, extracted_data)
    data         = _build_data(parsed, claim_amount, extracted_data)

    diagnosis_code = (
        data.get("diagnosis_code") or
        extracted_data.get("DiagnosisCode") or
        extracted_data.get("diagnosis_code") or ""
    )

    items = extract_itemized_bill(raw_text, diagnosis_code) if raw_text else []

    saved_count = 0
    if db_knowledge and items:
        for item in items:
            _, action = upsert_knowledge_item(db_knowledge, item, diagnosis_code, "ocr_extracted")
            if action in ("created", "updated"):
                saved_count += 1

    similar     = vector_search(db_knowledge, str(items or extracted_data), diagnosis_code) if db_knowledge else []
    rag_context = build_rag_context(items, similar)

    fraud_result       = predict_fraud(data)
    shap_contributions = explain_decision(data, fraud_result)
    shap_prompt        = format_shap_for_prompt(shap_contributions)

    decision_result = get_decision(
        query=extracted_data,
        rag_context=rag_context,
        itemized_items=items,
        fraud_result=fraud_result,
        shap_context=shap_prompt,
    )

    _save_claim(db_claims, data, decision_result, raw_text)

    return {
        "status":              "ok",
        "raw_text_preview":    raw_text[:2000] if raw_text else None,
        "extracted_data":      extracted_data,
        "itemized_items":      items,
        "items_count":         len(items),
        "saved_to_knowledge":  saved_count,
        "diagnosis_code":      diagnosis_code,
        "rag_references_used": len(similar),
        "fraud_score":         fraud_result["fraud_score"],
        "fraud_label":         fraud_result["label"],
        "fraud_flags":         fraud_result["flags"],
        "shap_factors":        shap_contributions,
        **decision_result,
    }


def _save_claim(db_claims, data: dict, decision_result: dict, raw_ocr_text: str = "") -> bool:
    if not db_claims:
        return False
    try:
        from ..models.models_user import Claim
        claim = Claim(
            claim_id                  = str(uuid.uuid4()),
            patient_age               = data.get("age"),
            patient_gender            = data.get("gender"),
            patient_income            = data.get("patient_income"),
            patient_employment_status = data.get("patient_employment"),
            patient_marital_status    = data.get("patient_marital"),
            policy_number             = data.get("policy_number"),
            date_of_service           = data.get("date_of_service"),
            hospital_name             = data.get("hospital_name"),
            pre_authorization_status  = data.get("pre_auth"),
            claim_type                = data.get("claim_type"),
            diagnosis_code            = data.get("diagnosis_code"),
            procedure_code            = data.get("procedure_code"),
            claim_amount              = data.get("claim_amount"),
            claim_status              = decision_result.get("decision", "Pending"),
            reimbursement_amount      = decision_result.get("reimbursement_amount", 0.0),
            decision                  = decision_result.get("decision"),
            confidence                = decision_result.get("confidence"),
            raw_ocr_text              = raw_ocr_text[:10000] if raw_ocr_text else None,
        )
        db_claims.add(claim)
        db_claims.commit()
        return True
    except Exception as e:
        print(f"[claim_service] _save_claim error: {e}")
        db_claims.rollback()
        return False