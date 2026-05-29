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

    sections = []

    if decision == "Approved":
        headline = "Your claim has been approved."
        sub = "We're pleased to let you know that after a careful and fair review, your claim has been approved. We've made this decision as transparent as possible for you."
    elif decision == "Partially Approved":
        headline = "Your claim has been partially approved."
        sub = "We reviewed your claim thoroughly and approved as much as your policy allows. Here's a clear explanation of our decision."
    else:
        headline = "Your claim has been approved."
        sub = "We're pleased to let you know that after a careful and fair review, your claim has been approved."

    sections.append(f"""
<div class="exp-container">
    <div class="exp-header">
        <div class="exp-headline">{headline}</div>
        <div class="exp-sub">{sub}</div>
    </div>
""")

    if final_amount > 0:
        sections.append(f"""
    <div class="exp-amounts">
        <div class="exp-amount-row">
            <span class="exp-label">Total Medical Bill</span>
            <span class="exp-value">${claim:,.2f}</span>
        </div>
        <div class="exp-amount-row covered">
            <span class="exp-label">Insurance Pays</span>
            <span class="exp-value">${final_amount:,.2f}</span>
        </div>
        <div class="exp-amount-row owe">
            <span class="exp-label">You Pay</span>
            <span class="exp-value">${you_pay:,.2f}</span>
        </div>
    </div>
""")

    why_items = []
    if pre_auth == "Yes":
        why_items.append("You took the important step of getting pre-authorization before treatment. This helped us verify your claim quickly and approve it with confidence.")

    if age >= 65:
        why_items.append(f"Since you are <strong>{age} years old</strong>, we were able to apply additional senior coverage benefits in your favor.")

    if why_items:
        items_html = "".join(f"<li>{item}</li>" for item in why_items)
        sections.append(f"""
    <div class="exp-section">
        <div class="exp-section-title">Why We Approved Your Claim</div>
        <ul class="exp-list">{items_html}</ul>
    </div>
""")

    sections.append(f"""
    <div class="exp-section">
        <div class="exp-section-title">How We Calculated Your Reimbursement</div>
        <div class="exp-calc">
            Your insurance plan covers <strong>{cov_pct}%</strong> of eligible medical expenses. We applied this rate fairly based on your policy terms.<br><br>
            <div class="exp-formula">
                ${claim:,.2f} × {cov_pct}% = <strong>${final_amount:,.2f}</strong>
            </div>
        </div>
    </div>
""")

    if shap_factors:
        notable = [f for f in shap_factors[:5] if abs(f.get("contribution", 0)) >= 0.05]
        if notable:
            rows = []
            for f in notable:
                feature = f.get("feature", "")
                label = feature.replace("_", " ").title()
                direction = f.get("direction", "")
                badge = "<span class='exp-badge green'>Positive Factor</span>" if direction == "decreases_risk" else "<span class='exp-badge orange'>Considered Factor</span>"
                
                explanation = ""
                if feature == "pre_auth":
                    explanation = "Having pre-authorization strongly supports your claim."
                elif feature == "claim_amount":
                    explanation = "The size of your bill was considered in the coverage calculation."
                elif feature == "amount_income_ratio":
                    explanation = "Your bill compared to your income level was taken into account."
                elif feature == "age":
                    explanation = "Your age affects the coverage level you qualify for."
                else:
                    explanation = "This factor was reviewed during the decision process."

                rows.append(f"""
                <div class='exp-factor-row'>
                    <span class='exp-factor-label'><strong>{label}:</strong> {explanation}</span>
                    {badge}
                </div>
                """)

            sections.append(f"""
    <div class="exp-section">
        <div class="exp-section-title">Key Factors Our AI System Considered</div>
        <p style="color:#555; margin-bottom:16px;">To make a fair decision, our system carefully analyzed several important pieces of information about your claim:</p>
        <div class="exp-factors">{"".join(rows)}</div>
    </div>
""")

    if pre_auth == "No" and decision not in ("Denied", "Pending"):
        sections.append("""
    <div class="exp-tip">
        <strong>A friendly tip for next time:</strong> Requesting pre-authorization before scheduled treatments can help you receive the maximum coverage possible and avoid unexpected costs.
    </div>
""")

    sections.append("</div>")
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
        fraud_block = f"\nFRAUD RISK: HIGH (score {fraud_result.get('fraud_score', 0):.3f}). Flags: {flags}.\n"

    prompt = f"""You are a strict health insurance claims adjudicator. 
You MUST return valid JSON with REAL NUMBERS only. Never use math expressions.

CLAIM:
- Amount: ${data.get('claim_amount', 0):,.2f}
- Pre-Authorization: {data.get('pre_auth', 'No')}
- Baseline: ${baseline:,.2f}
- Coverage Rate: {breakdown.get("coverage_pct", 75)}%

RULES (follow strictly):
1. Pre-Auth = "Yes" → Decision = "Approved"
2. Pre-Auth = "No" and Amount > 5000 → Decision = "Denied", reimbursement_amount = 0
3. reimbursement_amount and eligible_amount MUST be real numbers (e.g. 84.38), NOT expressions like "112.50 * 0.75"
4. Return only valid JSON, no extra text.

Return exactly this format:

{{
  "decision": "Approved" | "Partially Approved" | "Denied" | "Pending",
  "reimbursement_amount": number,
  "eligible_amount": number,
  "reason": "short_snake_case_reason",
  "flagged_items": [],
  "policy_type": "standard",
  "confidence": 0.80
}}"""

    try:
        res = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            temperature=0.0,
            max_tokens=700,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        
        content = res.choices[0].message.content.strip()
        result = safe_json(content)

        if isinstance(result, dict) and result.get("decision"):
            # Force number for reimbursement_amount
            try:
                if isinstance(result.get("reimbursement_amount"), (str,)):
                    result["reimbursement_amount"] = float(result["reimbursement_amount"])
                else:
                    result["reimbursement_amount"] = float(result.get("reimbursement_amount", baseline * 0.75))
            except:
                result["reimbursement_amount"] = round(baseline * 0.75, 2)

            # Confidence
            raw_conf = float(result.get("confidence", 0.78))
            result["confidence"] = round(min(max(raw_conf, 0.65), 0.95), 2)

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
            "explanation": "We could not process your claim yet because some required information is missing.",
            "confidence": 0.0,
            "needs_human_review": True,
            "risk_level": "High",
        }

    claim_amount        = _get_amount(parsed, raw)
    data                = _build_data(parsed, claim_amount, raw)
    baseline, breakdown = calculate_baseline_reimbursement(claim_amount, data)

    llm_result = llm_calculate_reimbursement(
        data, baseline, breakdown, itemized_items or [], rag_context, fraud_result
    )

    decision     = llm_result.get("decision", "Pending")
    final_amount = float(llm_result.get("reimbursement_amount", baseline))
    confidence   = llm_result.get("confidence", 0.75)
    fraud_score  = fraud_result.get("fraud_score", 0.0) if fraud_result else 0.0

    risk_level = "Low"
    needs_human_review = False

    if fraud_score >= 0.75 or confidence < 0.70 or claim_amount > 20000:
        risk_level = "High"
        needs_human_review = True
        decision = "Pending" 
    elif (fraud_score >= 0.45 or confidence < 0.85 or 
          (data["pre_auth"] == "No" and claim_amount > 8000)):
        risk_level = "Medium"
        needs_human_review = True
    else:
        risk_level = "Low"
        needs_human_review = False

    if needs_human_review and decision == "Approved":
        decision = "Pending"

    if data["pre_auth"] == "No" and claim_amount > PRE_AUTH_THRESHOLD:
        final_amount = 0.0
        decision = "Denied"
        needs_human_review = True
        risk_level = "High"

    elif baseline > 0:
        min_allowed  = baseline * 0.65
        max_allowed  = max(baseline * 1.35, claim_amount)
        final_amount = max(min_allowed, min(max_allowed, final_amount))

    explanation = _build_human_explanation(
        data, breakdown, decision, final_amount, shap_factors=shap_factors or [],
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
        "confidence":           round(confidence, 2),
        "risk_level":           risk_level,           # New
        "needs_human_review":   needs_human_review,   # New
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