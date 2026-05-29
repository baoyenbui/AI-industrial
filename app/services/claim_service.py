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
from .shap_service import explain_decision

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
    candidates = [
        parsed.get("amount"),
        parsed.get("claimamount"),
        raw.get("claim_amount"),
        raw.get("ClaimAmount"),
        raw.get("claimAmount"),
        raw.get("claimamount"),
    ]
    for v in candidates:
        if v is None:
            continue
        result = safe_float(v)
        if result is not None and result > 0:
            return result
    return 0.0


def _get_pre_auth_raw(raw: dict) -> str:
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


def calculate_baseline_reimbursement(claim_amount: float, data: dict) -> tuple[float, dict]:
    if claim_amount <= 0:
        return 0.0, {"claim_amount": 0, "coverage_pct": 0, "baseline": 0.0, "copay": 0.0, "deductible_not_met": False}

    coverage_pct = 0.75

    if data.get("age", 0) >= 65:
        coverage_pct = 0.85

    diagnosis = data.get("diagnosis_code", "")
    if any(code in diagnosis for code in HIGH_RISK_CODES):
        coverage_pct = 0.70

    if data.get("pre_auth") == "No" and claim_amount > PRE_AUTH_THRESHOLD:
        coverage_pct = max(0.10, coverage_pct - 0.15)

    reimbursement = round(claim_amount * coverage_pct, 2)

    breakdown = {
        "claim_amount":       claim_amount,
        "coverage_pct":       int(coverage_pct * 100),
        "baseline":           reimbursement,
        "copay":              0.0,
        "deductible_not_met": False,
    }
    return reimbursement, breakdown

def _build_human_explanation(
    data: dict,
    breakdown: dict,
    decision: str,
    final_amount: float,
    shap_factors: list = None,
) -> str:
    claim = float(breakdown.get("claim_amount", 0))
    cov_pct = int(breakdown.get("coverage_pct", 75))
    pre_auth = data.get("pre_auth", "No")
    age = int(data.get("age") or 0)
    you_pay = round(claim - final_amount, 2) if final_amount > 0 else claim
    diag = data.get("diagnosis_code", "")

    sections = []

    # Header
    if decision == "Approved":
        headline = "Your claim has been approved."
        sub = "Here is a breakdown of what your insurance covers and what you owe."
    elif decision == "Partially Approved":
        headline = "Your claim has been partially approved."
        sub = "Your insurance will cover part of the cost. See the breakdown below."
    elif decision == "Denied":
        headline = "Your claim was not approved."
        sub = "Please read below to understand why and what you can do next."
    else:
        headline = "Your claim is under review."
        sub = "We need a bit more information before a final decision can be made."

    sections.append(f"""
<div class="exp-container">
    <div class="exp-header">
        <div class="exp-headline">{headline}</div>
        <div class="exp-sub">{sub}</div>
    </div>
""")

    # Amounts
    if decision not in ("Denied", "Pending") and final_amount > 0:
        sections.append(f"""
    <div class="exp-amounts">
        <div class="exp-amount-row">
            <span class="exp-label">Total medical bill</span>
            <span class="exp-value">${claim:,.2f}</span>
        </div>
        <div class="exp-amount-row covered">
            <span class="exp-label">Insurance pays</span>
            <span class="exp-value">${final_amount:,.2f}</span>
        </div>
        <div class="exp-amount-row owe">
            <span class="exp-label">You pay</span>
            <span class="exp-value">${you_pay:,.2f}</span>
        </div>
    </div>
""")

    # Why this decision
    why_items = []
    if pre_auth == "Yes":
        why_items.append("You obtained pre-authorization before treatment. This is the most important step to ensure your claim is covered.")
    elif pre_auth == "No" and claim > 5000.0:
        why_items.append(f"Your bill of <strong>${claim:,.2f}</strong> exceeds the $5,000 threshold that requires prior approval. No pre-authorization was obtained.")
    elif pre_auth == "No":
        why_items.append("No pre-authorization was on file. Your coverage rate has been slightly reduced as per policy terms.")

    if age >= 65:
        why_items.append(f"As a patient aged <strong>{age}</strong>, you qualify for enhanced senior coverage ({cov_pct}%).")

    if any(code in diag for code in {"C50", "C61", "C34", "E11", "I21", "I25"}):
        why_items.append(f"Your diagnosis code <strong>{diag}</strong> is a high-priority condition.")

    if why_items:
        items_html = "".join(f"<li>{item}</li>" for item in why_items)
        sections.append(f"""
    <div class="exp-section">
        <div class="exp-section-title">Why this decision was made</div>
        <ul class="exp-list">{items_html}</ul>
    </div>
""")

    # Calculation
    if decision not in ("Denied", "Pending") and cov_pct > 0 and final_amount > 0:
        sections.append(f"""
    <div class="exp-section">
        <div class="exp-section-title">How your reimbursement was calculated</div>
        <div class="exp-calc">
            Your plan covers <strong>{cov_pct}%</strong> of eligible costs.<br><br>
            <div class="exp-formula">${claim:,.2f} × {cov_pct}% = <strong>${final_amount:,.2f}</strong></div>
        </div>
    </div>
""")

    # Key factors
    if shap_factors:
        FRIENDLY = {
            "claim_amount": "Size of your medical bill",
            "pre_auth": "Prior approval from your insurer",
            "amount_income_ratio": "Bill size relative to your income",
            "age": "Patient age",
            "diagnosis_code": "Type of medical condition",
            "patient_income": "Reported monthly income",
            "claim_type": "Type of treatment",
            "provider_specialty": "Medical specialty of your doctor",
            "submission_method": "How the claim was submitted",
        }
        notable = [f for f in shap_factors[:5] if abs(f.get("contribution", 0)) >= 0.05]
        if notable:
            rows = []
            for f in notable:
                label = FRIENDLY.get(f.get("feature", ""), f.get("feature", "").replace("_", " ").title())
                direction = f.get("direction", "")
                badge = "<span class='exp-badge green'>Helped your claim</span>" if direction == "decreases_risk" else "<span class='exp-badge orange'>Reduced coverage</span>"
                rows.append(f"<div class='exp-factor-row'><span class='exp-factor-label'>{label}</span>{badge}</div>")

            sections.append(f"""
    <div class="exp-section">
        <div class="exp-section-title">Key factors considered</div>
        <div class="exp-factors">{"".join(rows)}</div>
    </div>
""")

    # Next steps / Tip
    if decision == "Denied" and pre_auth == "No" and claim > 5000.0:
        sections.append("""
    <div class="exp-section exp-next-steps">
        <div class="exp-section-title">What you can do now</div>
        <ol class="exp-steps">
            <li><strong>File an appeal</strong> — Contact your insurer to appeal.</li>
            <li><strong>Request retroactive authorization</strong> — For urgent cases.</li>
            <li><strong>Ask your doctor's office for help</strong> — Supporting documents.</li>
            <li><strong>Call the number on your insurance card</strong> for appeals process.</li>
        </ol>
    </div>
""")
    elif pre_auth == "No" and decision not in ("Denied", "Pending"):
        sections.append("""
    <div class="exp-tip">
        <strong>Tip for next time:</strong> Always request pre-authorization before non-emergency procedures.
    </div>
""")

    sections.append("</div>")  # Close container

    return "".join(sections).strip()

def llm_calculate_reimbursement(
    data: dict,
    baseline: float,
    breakdown: dict,
    itemized_items: list = None,
    rag_context: str = "",
    fraud_result: dict = None,
) -> dict:
    items_block = json.dumps(itemized_items, indent=2) if itemized_items else "No itemized data"
    item_count  = len(itemized_items) if itemized_items else 0

    fraud_block = ""
    if fraud_result and fraud_result.get("label") == "HIGH_RISK":
        flags = ", ".join(fraud_result.get("flags", [])) or "none"
        fraud_block = (
            f"\nFRAUD RISK: HIGH (score {fraud_result.get('fraud_score', 0):.3f}). "
            f"Flags: {flags}. Reduce confidence and consider Partial Approval.\n"
        )

    prompt = f"""You are a health insurance claims adjudicator. Return a JSON decision.

CLAIM:
- Amount:            ${data.get('claim_amount', 0):,.2f}
- Pre-Authorization: {data.get('pre_auth', 'No')}
- Patient Age:       {data.get('age', 'N/A')}
- Diagnosis Code:    {data.get('diagnosis_code') or 'N/A'}
- Procedure Code:    {data.get('procedure_code') or 'N/A'}
- Specialty:         {data.get('provider_specialty') or 'N/A'}
- Hospital:          {data.get('hospital_name') or 'N/A'}
- Rule Baseline:     ${baseline:,.2f}
- Coverage Rate:     {breakdown.get("coverage_pct", 75)}%
{fraud_block}
ITEMIZED BILL ({item_count} items):
{items_block}

KNOWLEDGE BASE:
{rag_context or 'No reference data.'}

RULES (apply in order):
1. Missing Policy Number does NOT affect the decision.
2. Missing Diagnosis Code → decision must be "Pending".
3. Pre-Auth=Yes → Approved.
4. Pre-Auth=No AND Amount > ${PRE_AUTH_THRESHOLD:,.0f} → Denied, reimbursement=0.
5. Pre-Auth=No AND Amount <= ${PRE_AUTH_THRESHOLD:,.0f} → Approved at 70% of baseline.
6. If coverage rate is low and claim is high, consider Partial Approval.
7. Flag any line item priced more than 30% above knowledge base average.
8. Confidence must reflect actual data completeness. Never return 1.0.

Return ONLY valid JSON:
{{
  "decision": "Approved" | "Partially Approved" | "Denied" | "Pending",
  "reimbursement_amount": number,
  "eligible_amount": number,
  "reason": "short_snake_case_reason",
  "flagged_items": [],
  "policy_type": "standard" | "premium" | "high_risk",
  "confidence": 0.00
}}"""

    try:
        res = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            temperature=0.0,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        result = safe_json(res.choices[0].message.content)
        if isinstance(result, dict) and result.get("decision"):
            raw_conf = float(result.get("confidence", 0.75))
            raw_conf = min(raw_conf, 0.95)
            if not (data.get("diagnosis_code") and data.get("procedure_code") and data.get("hospital_name")):
                raw_conf = min(raw_conf, 0.82)
            result["confidence"] = round(raw_conf, 2)
            return result
    except Exception as e:
        print(f"[claim_service] LLM error: {e}")

    return _fallback_decision(data, baseline)


def _fallback_decision(data: dict, baseline: float) -> dict:
    pre_auth = data.get("pre_auth", "No")
    amount   = float(data.get("claim_amount", 0))

    if not data.get("diagnosis_code"):
        return {
            "decision": "Pending", "reimbursement_amount": 0.0, "eligible_amount": 0.0,
            "reason": "missing_diagnosis_code", "flagged_items": [],
            "policy_type": "standard", "confidence": 0.60,
        }

    if pre_auth == "Yes":
        reimburse = round(baseline * 0.85, 2)
        return {
            "decision": "Approved", "reimbursement_amount": reimburse, "eligible_amount": reimburse,
            "reason": "pre_authorization_confirmed", "flagged_items": [],
            "policy_type": "standard", "confidence": 0.82,
        }

    if amount > PRE_AUTH_THRESHOLD:
        return {
            "decision": "Denied", "reimbursement_amount": 0.0, "eligible_amount": 0.0,
            "reason": "no_pre_auth_exceeds_threshold", "flagged_items": [],
            "policy_type": "standard", "confidence": 0.90,
        }

    reduced = round(baseline * 0.70, 2)
    return {
        "decision": "Approved", "reimbursement_amount": reduced, "eligible_amount": reduced,
        "reason": "no_pre_auth_under_threshold", "flagged_items": [],
        "policy_type": "standard", "confidence": 0.75,
    }


def get_decision(
    query=None,
    rag_context: str = "",
    itemized_items: list = None,
    fraud_result: dict = None,
    shap_factors: list = None,
) -> dict:
    raw    = query if isinstance(query, dict) else {}
    parsed = parse_query(query)

    missing = detect_missing(parsed)
    if missing:
        return {
            "status": "ok", "decision": "Pending", "reason": "missing_required_fields",
            "missing_fields": missing, "reimbursement_amount": 0.0, "baseline_amount": 0.0,
            "explanation": (
                "We could not process your claim yet because the following information is missing: "
                + ", ".join(missing) + ". Please fill in all required fields and resubmit."
            ),
            "confidence": 0.0,
        }

    claim_amount        = _get_amount(parsed, raw)
    data                = _build_data(parsed, claim_amount, raw)
    baseline, breakdown = calculate_baseline_reimbursement(claim_amount, data)

    llm_result = llm_calculate_reimbursement(
        data, baseline, breakdown, itemized_items or [], rag_context, fraud_result
    )

    decision     = llm_result.get("decision", "Pending")
    final_amount = float(llm_result.get("reimbursement_amount", baseline))

    if data["pre_auth"] == "No" and claim_amount > PRE_AUTH_THRESHOLD:
        final_amount = 0.0
        decision     = "Denied"
    elif baseline > 0:
        min_allowed  = baseline * 0.65
        max_allowed  = max(baseline * 1.35, claim_amount)
        final_amount = max(min_allowed, min(max_allowed, final_amount))

    explanation = _build_human_explanation(
        data, breakdown, decision, final_amount,
        shap_factors=shap_factors or [],
    )

    return {
        "status":               "ok",
        "decision":             decision,
        "reason":               llm_result.get("reason", "processed"),
        "reimbursement_amount": round(final_amount, 2),
        "baseline_amount":      round(baseline, 2),
        "adjustment_percent":   round((final_amount / baseline - 1) * 100, 1) if baseline > 0 else 0,
        "explanation":          explanation,
        "flagged_items":        llm_result.get("flagged_items", []),
        "policy_type":          llm_result.get("policy_type", "standard"),
        "confidence":           llm_result.get("confidence", 0.75),
        "policy_number":        data["policy_number"],
        "date_of_service":      data["date_of_service"],
        "hospital_name":        data["hospital_name"],
        "pre_auth":             data["pre_auth"],
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
    parsed         = parse_query(extracted_data)
    claim_amount   = _get_amount(parsed, extracted_data)
    data           = _build_data(parsed, claim_amount, extracted_data)

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

    decision_result = get_decision(
        query=extracted_data,
        rag_context=rag_context,
        itemized_items=items,
        fraud_result=fraud_result,
        shap_factors=shap_contributions,
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
        "fraud_score":         fraud_result.get("fraud_score", 0),
        "fraud_label":         fraud_result.get("label", ""),
        "fraud_flags":         fraud_result.get("flags", []),
        "shap_factors":        shap_contributions,
        **decision_result,
    }