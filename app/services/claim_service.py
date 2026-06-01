import uuid
import json
import os
import sqlite3
from html import escape
from groq import Groq
from dotenv import load_dotenv
from ..utils import safe_float, safe_int, safe_json, parse_query
from .extraction_service import extract_claim_data
from .ocr_itemized import extract_itemized_bill
from .rag_service import vector_search, build_rag_context
from .knowledge_service import upsert_knowledge_item
from .fraud_service import predict_fraud
from .shap_service import explain_decision

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

HIGH_RISK_CODES = {"C50", "C61", "C34", "E11", "I21", "I25"}
PRE_AUTH_THRESHOLD = 5000.0
REQUIRED_FORM_FIELDS = [
    "PatientAge",
    "PatientGender",
    "PatientEmploymentStatus",
    "ClaimAmount",
    "DiagnosisCode",
    "ProviderName",
    "PreAuthorizationStatus",
    "ProcedureCode",
    "ClaimSubmissionMethod",
    "ProviderSpecialty",
    "ClaimType",
    "PatientMaritalStatus",
    "PolicyNumber",
]

def _resolve_pre_auth(parsed: dict) -> str:
    for k in ["preauthorizationstatus", "preauthstatus", "preauthorization", "preauth", "pre_auth", "PreAuthorizationStatus", "PreAuthStatus"]:
        v = parsed.get(k)
        if v is not None:
            s = str(v).strip().lower()
            if s in ("yes", "true", "1", "approved", "granted"):
                return "Yes"
            if s in ("no", "false", "0", "denied", "not required"):
                return "No"
    return "No"

def _get_amount(parsed: dict, raw: dict) -> float:
    keys = ["ClaimAmount", "claim_amount", "claimamount", "amount", "claimAmount"]
    for k in keys:
        v = raw.get(k)
        if v is not None:
            result = safe_float(v)
            if result is not None and result > 0:
                return result
    for k in keys:
        v = parsed.get(k)
        if v is not None:
            result = safe_float(v)
            if result is not None and result > 0:
                return result
    return 60.0

def _get_pre_auth_raw(raw: dict) -> str:
    for k in ["PreAuthorizationStatus", "pre_authorization_status", "preauthorizationstatus", "pre_auth_status", "PreAuthStatus", "pre_auth"]:
        v = raw.get(k)
        if v is not None:
            s = str(v).strip().lower()
            if s in ("yes", "true", "1", "approved", "granted"):
                return "Yes"
            if s in ("no", "false", "0", "denied", "not required"):
                return "No"
    return "No"

def _build_data(parsed: dict, claim_amount: float, raw: dict = None) -> dict:
    raw = raw or {}
    pre_auth = _resolve_pre_auth(parsed)
    if not pre_auth or pre_auth == "No":
        raw_pre = _get_pre_auth_raw(raw)
        if raw_pre:
            pre_auth = raw_pre
    return {
        "claim_amount": claim_amount,
        "age": safe_int(raw.get("PatientAge") or parsed.get("patient_age") or parsed.get("PatientAge") or 30) or 30,
        "gender": str(raw.get("PatientGender") or parsed.get("patient_gender") or parsed.get("PatientGender") or "Other"),
        "claim_type": str(raw.get("ClaimType") or parsed.get("claim_type") or parsed.get("ClaimType") or "medical").lower(),
        "diagnosis_code": str(raw.get("DiagnosisCode") or parsed.get("diagnosis_code") or parsed.get("DiagnosisCode") or "Z00.00"),
        "procedure_code": str(raw.get("ProcedureCode") or parsed.get("procedure_code") or parsed.get("ProcedureCode") or "99213"),
        "provider_specialty": str(raw.get("ProviderSpecialty") or parsed.get("provider_specialty") or parsed.get("ProviderSpecialty") or "general").lower(),
        "patient_income": safe_float(raw.get("PatientIncome") or parsed.get("patient_income") or parsed.get("PatientIncome")) or 3000.0,
        "patient_employment": str(raw.get("PatientEmploymentStatus") or parsed.get("patient_employment") or parsed.get("PatientEmploymentStatus") or "unknown"),
        "patient_marital": str(raw.get("PatientMaritalStatus") or parsed.get("patient_marital") or parsed.get("PatientMaritalStatus") or "single"),
        "policy_number": str(raw.get("PolicyNumber") or parsed.get("policy_number") or parsed.get("PolicyNumber") or ""),
        "date_of_service": str(raw.get("DateOfService") or parsed.get("date_of_service") or parsed.get("DateOfService") or ""),
        "hospital_name": str(raw.get("ProviderName") or parsed.get("provider_name") or parsed.get("ProviderName") or "Unknown Provider"),
        "pre_auth": pre_auth,
        "submission_method": str(raw.get("ClaimSubmissionMethod") or parsed.get("claim_submission_method") or parsed.get("ClaimSubmissionMethod") or "online"),
        "insurance_company": str(raw.get("insurance_company") or raw.get("InsuranceCompany") or "Default Insurance"),
    }

def get_company_rules(company_name: str) -> dict:
    if not company_name:
        return {"company_id": "DEFAULT", "company_name": "Default Insurance", "default_reimbursement_percent": 0.80, "coverage_rules": {}}
    try:
        from ..core.database_user import DB_PATH
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT company_id, company_name, coverage_rules, default_reimbursement_percent FROM insurance_companies WHERE company_name = ?", (company_name,))
        result = cursor.fetchone()
        conn.close()
        if result:
            return {
                "company_id": result[0],
                "company_name": result[1],
                "coverage_rules": json.loads(result[2]) if result[2] else {},
                "default_reimbursement_percent": result[3] or 0.80,
            }
        return {"company_id": "DEFAULT", "company_name": "Default Insurance", "default_reimbursement_percent": 0.80, "coverage_rules": {}}
    except Exception:
        return {"company_id": "DEFAULT", "company_name": "Default Insurance", "default_reimbursement_percent": 0.80, "coverage_rules": {}}

def calculate_dynamic_reimbursement(claim_amount: float, data: dict) -> tuple:
    claim_amount = max(0.01, float(claim_amount or 0.0))
    company_name = data.get("insurance_company", "Default Insurance")
    company_rules = get_company_rules(company_name)
    default_percent = company_rules["default_reimbursement_percent"]
    rules = company_rules["coverage_rules"]
    claim_type = data.get("claim_type", "medical")
    pre_auth = data.get("pre_auth", "No")
    age = data.get("age", 30)
    diagnosis = data.get("diagnosis_code", "Z00.00")
    applied_percent = default_percent
    rule_applied = "standard_policy"
    rule_label = "standard policy"
    max_amount = None
    for rule_id, rule in rules.items():
        condition = str(rule.get("condition", "")).strip().lower()
        try:
            condition_check = True
            if "claim_type ==" in condition:
                expected_type = condition.split("claim_type ==")[1].strip().replace("'", "").replace('"', "")
                condition_check = condition_check and (claim_type == expected_type.lower())
            if "pre_auth ==" in condition:
                expected_auth = condition.split("pre_auth ==")[1].strip().replace("'", "").replace('"', "")
                condition_check = condition_check and (pre_auth == expected_auth)
            if "age >=" in condition:
                min_age = int(condition.split("age >=")[1].split()[0])
                condition_check = condition_check and (age >= min_age)
            if any(code in diagnosis for code in HIGH_RISK_CODES) and "high_risk" in condition:
                condition_check = condition_check and True
            if condition_check:
                applied_percent = rule.get("reimbursement_percent", default_percent)
                rule_applied = rule_id
                rule_label = rule.get("label") or rule.get("name") or rule_id
                max_amount = rule.get("max_amount")
                break
        except Exception:
            continue
    deductible = 30.0 if age >= 65 else (80.0 if any(code in diagnosis for code in HIGH_RISK_CODES) else 50.0)
    eligible = claim_amount
    after_deduct = max(0.0, eligible - deductible)
    baseline = round(after_deduct * applied_percent, 2)
    reimbursement = baseline
    if max_amount and reimbursement > max_amount:
        reimbursement = max_amount
    if reimbursement <= 0:
        reimbursement = round((claim_amount - 50.0) * 0.80, 2)
        if reimbursement < 0:
            reimbursement = claim_amount * 0.70
    if reimbursement <= 0:
        reimbursement = max(round(claim_amount * 0.70, 2), 10.0)
    breakdown = {
        "claim_amount": claim_amount,
        "eligible": eligible,
        "deductible": deductible,
        "after_deduct": after_deduct,
        "coverage_pct": int(applied_percent * 100),
        "copay": 0,
        "baseline": baseline,
        "deductible_not_met": after_deduct == 0,
        "reimbursement_percent": applied_percent,
        "rule_applied": rule_applied,
        "rule_label": rule_label,
        "max_amount": max_amount,
        "company_id": company_rules["company_id"],
        "company_name": company_name,
    }
    return reimbursement, breakdown, rule_applied

def _shap_groups(shap_factors: list = None) -> dict:
    groups = {
        "decision_summary": [],
        "policy_context": [],
        "clinical_context": [],
        "model_signals": [],
        "knowledge_context": [],
        "future": [],
    }
    if not shap_factors:
        return groups
    for x in shap_factors:
        feat = str(x.get("feature", "")).lower()
        expl = str(x.get("explanation", "")).strip()
        if feat in ("pre_auth", "preauthorization", "preauthorizationstatus"):
            groups["policy_context"].append(expl)
        elif feat in ("claim_amount", "amount_income_ratio", "distance_to_provider", "claim_frequency_30d"):
            groups["model_signals"].append(expl)
        elif feat in ("drug_disease_match", "diagnosis_procedure_consistency"):
            groups["clinical_context"].append(expl)
        elif feat in ("provider_history_score",):
            groups["knowledge_context"].append(expl)
        elif feat in ("age", "gender"):
            groups["clinical_context"].append(expl)
        else:
            if expl:
                groups["model_signals"].append(expl)
    for k in groups:
        groups[k] = [v for i, v in enumerate(groups[k]) if v and v not in groups[k][:i]]
    return groups

def _tone_word(decision: str) -> tuple[str, str]:
    if decision == "Approved":
        return "Approved", "Your claim is covered."
    if decision == "Denied":
        return "Not approved", "This claim could not be covered based on the information provided."
    return "Under review", "We still need a bit more information before a final answer."

def _clean_sentences(items):
    out = []
    for item in items:
        text = str(item).strip()
        if text:
            if text[-1] not in ".!?":
                text += "."
            out.append(text)
    return out

def _friendly_reason(text: str) -> str:
    t = str(text or "").strip()
    if not t:
        return ""
    if t.startswith("Good sign:"):
        return t
    if t.startswith("Clinical fit:"):
        return t
    if t.startswith("Procedure fit:"):
        return t
    if t.startswith("Provider context:"):
        return t
    if t.startswith("Specialty context:"):
        return t
    if t.startswith("Policy path:"):
        return ""
    if "pre-authorization" in t.lower():
        return "Good sign: pre-authorization was already in place."
    if "diagnosis" in t.lower():
        return "Clinical fit: the diagnosis matched the expected profile."
    if "procedure" in t.lower():
        return "Procedure fit: the submitted procedure matched the claim."
    if "provider" in t.lower():
        return "Provider context: the provider details were incomplete, so the claim details mattered more."
    if "specialty" in t.lower():
        return "Specialty context: the provider specialty can affect how similar claims are handled."
    if "policy" in t.lower():
        return ""
    return t

def _confidence_copy(confidence: float) -> str:
    c = max(0.0, min(100.0, float(confidence or 0.0)))
    if c <= 30:
        return f"Confidence is {c:.2f}%, so this result should be treated with caution."
    if c <= 60:
        return f"Confidence is {c:.2f}%, because the main details are present, but a few parts still need care."
    if c <= 80:
        return f"Confidence is {c:.2f}%, because most of the key details are available and consistent."
    return f"Confidence is {c:.2f}%, because the claim has clear, complete details that support the result."

def _format_money(amount: float) -> str:
    return f"${amount:,.2f}"

def _collect_explanation_sections(data: dict, breakdown: dict, decision: str, final_amount: float, confidence: float = 0.0, shap_factors: list = None, knowledge_reference: str = None) -> dict:
    claim = float(breakdown.get("claim_amount", 60))
    deductible = float(breakdown.get("deductible", 50))
    after_deduct = float(breakdown.get("after_deduct", 0))
    cov_pct = int(breakdown.get("coverage_pct", 80))
    baseline = float(breakdown.get("baseline", final_amount))
    pre_auth = data.get("pre_auth", "No")
    age = int(data.get("age") or 30)
    diagnosis = data.get("diagnosis_code", "Z00.00")
    procedure_code = data.get("procedure_code", "99213")
    provider = data.get("hospital_name", "Unknown Provider")
    specialty = data.get("provider_specialty", "general")
    claim_type = data.get("claim_type", "medical")
    company_name = breakdown.get("company_name", "Default Insurance")
    rule_label = breakdown.get("rule_label", breakdown.get("rule_applied", "standard policy"))
    amount_payable = round(final_amount, 2)
    out_of_pocket = round(max(claim - amount_payable, 0.0), 2)
    title, subtitle = _tone_word(decision)
    summary_items = [
        "Reimbursement means the amount insurance pays for this claim.",
        "Baseline means the starting payout from the policy rules before final adjustments.",
        _confidence_copy(confidence),
    ]
    calc_items = [
        f"Total bill: {_format_money(claim)}.",
        f"Deductible: - {_format_money(deductible)}.",
        f"After deductible: {_format_money(after_deduct)}.",
        f"Coverage: {cov_pct}%.",
        f"Final reimbursement: {_format_money(amount_payable)}.",
    ]
    why_items = []
    if pre_auth == "Yes":
        why_items.append("Good sign: pre-authorization was already in place.")
    else:
        why_items.append("No pre-authorization was found, so the claim followed the usual coverage path.")
    if any(code in diagnosis for code in HIGH_RISK_CODES):
        why_items.append(f"Clinical fit: the diagnosis code {diagnosis} falls into a higher-complexity group.")
    else:
        why_items.append("Clinical fit: the diagnosis matched the expected profile.")
    why_items.append(f"Procedure fit: procedure {procedure_code} matched the submitted treatment.")
    if provider and provider != "Unknown Provider":
        why_items.append(f"Provider context: {provider} was included in the review.")
    else:
        why_items.append("Provider context: the provider details were incomplete, so the claim details mattered more.")
    if specialty and specialty != "general":
        why_items.append(f"Specialty context: {specialty} claims can be handled differently depending on the plan.")
    groups = _shap_groups(shap_factors or [])
    if groups["model_signals"]:
        why_items.append("Model signal: the claim amount looked reasonable for the patient profile.")
    why_items = _clean_sentences([_friendly_reason(x) for x in why_items])
    improve_items = []
    if decision == "Denied":
        if pre_auth == "No":
            improve_items.append("Request pre-authorization before treatment for this type of claim.")
        if claim > PRE_AUTH_THRESHOLD:
            improve_items.append("Split large services into documented episodes only if the plan allows it.")
        if not provider or provider == "Unknown Provider":
            improve_items.append("Add the full provider or clinic name to reduce matching errors.")
        if not diagnosis or diagnosis == "Z00.00":
            improve_items.append("Include a specific diagnosis code that matches the treatment.")
        if not procedure_code or procedure_code == "99213":
            improve_items.append("Add the exact procedure code from the bill or medical record.")
        if not improve_items:
            improve_items.append("Add clearer medical and billing documents to support the claim.")
    else:
        if pre_auth == "No":
            improve_items.append("Future claims may process more smoothly with pre-authorization.")
        if provider == "Unknown Provider":
            improve_items.append("Include the full provider or clinic name next time.")
        if any(code in diagnosis for code in HIGH_RISK_CODES):
            improve_items.append("Attach supporting medical notes for higher-complexity diagnoses.")
        if not improve_items:
            improve_items.append("Keep diagnosis, procedure, and provider details complete for faster review.")
    return {
        "summary_items": summary_items,
        "calc_items": calc_items,
        "why_items": why_items,
        "improve_items": _clean_sentences(improve_items),
        "headline": title,
        "subheadline": subtitle,
        "claim": claim,
        "deductible": deductible,
        "after_deduct": after_deduct,
        "cov_pct": cov_pct,
        "amount_payable": amount_payable,
        "out_of_pocket": out_of_pocket,
        "baseline": baseline,
        "company_name": company_name,
        "rule_label": rule_label,
    }

def _format_li(items):
    return "".join(f"<li>{escape(str(x))}</li>" for x in items if x)

def _build_human_explanation(data: dict, breakdown: dict, decision: str, final_amount: float, confidence: float = 0.0, shap_factors: list = None, knowledge_reference: str = None) -> str:
    e = _collect_explanation_sections(
        data,
        breakdown,
        decision,
        final_amount,
        confidence=confidence,
        shap_factors=shap_factors,
        knowledge_reference=knowledge_reference,
    )
    return f'''<div class="exp-container">
  <div class="exp-header">
    <div class="exp-chip exp-chip-{escape(decision.lower())}">{escape(e["headline"])}</div>
    <div class="exp-headline">{escape(e["subheadline"])}</div>
  </div>

  <div class="exp-section exp-section-overview">
    <div class="exp-section-title">What this means</div>
    <ul class="exp-list">{_format_li(e["summary_items"])}</ul>
  </div>

  <div class="exp-section">
    <div class="exp-section-title">How we worked this out</div>
    <div class="exp-calc-table">
      <div class="exp-calc-row exp-calc-header">
        <span class="exp-calc-label" style="font-weight:700;color:#475569;">Total bill</span>
        <span class="exp-calc-value exp-color-neutral" style="font-weight:700;color:#475569;">{_format_money(e["claim"])}</span>
      </div>
      <div class="exp-calc-row exp-calc-deduction">
        <span class="exp-calc-label" style="font-weight:700;color:#f59e0b;">Deductible</span>
        <span class="exp-calc-value exp-color-warn" style="font-weight:700;color:#f59e0b;">- {_format_money(e["deductible"])}</span>
      </div>
      <div class="exp-calc-row exp-calc-subtle">
        <span class="exp-calc-label" style="font-weight:700;color:#ef4444;">After deductible</span>
        <span class="exp-calc-value exp-color-danger" style="font-weight:700;color:#ef4444;">{_format_money(e["after_deduct"])}</span>
      </div>
      <div class="exp-calc-row exp-calc-subtle">
        <span class="exp-calc-label" style="font-weight:700;color:#0ea5e9;">Coverage</span>
        <span class="exp-calc-value exp-color-info" style="font-weight:700;color:#0ea5e9;">{e["cov_pct"]}%</span>
      </div>
      <div class="exp-calc-row exp-calc-total">
        <span class="exp-calc-label" style="font-weight:700;color:#16a34a;">Final reimbursement</span>
        <span class="exp-calc-value exp-color-success" style="font-weight:700;color:#16a34a;">{_format_money(e["amount_payable"])}</span>
      </div>
    </div>
  </div>

  <div class="exp-section">
    <div class="exp-section-title">Why this result</div>
    <ul class="exp-list">{_format_li(e["why_items"])}</ul>
  </div>
  
  <div class="exp-section">
    <div class="exp-section-title">How to improve results</div>
    <ul class="exp-list">{_format_li(e["improve_items"])}</ul>
  </div>
</div>'''

def _fast_decision(data: dict, baseline: float) -> dict:
    pre_auth = data.get("pre_auth", "No")
    amount = float(data.get("claim_amount", 60))
    if pre_auth == "Yes":
        return {
            "decision": "Approved",
            "reimbursement_amount": round(baseline * 0.85, 2),
            "eligible_amount": round(baseline * 0.85, 2),
            "reason": "pre_authorization_confirmed",
            "flagged_items": [],
            "policy_type": "standard",
            "confidence": 82,
        }
    if amount > PRE_AUTH_THRESHOLD:
        return {
            "decision": "Denied",
            "reimbursement_amount": 0.0,
            "eligible_amount": 0.0,
            "reason": "no_pre_auth_exceeds_threshold",
            "flagged_items": [],
            "policy_type": "standard",
            "confidence": 90,
        }
    return {
        "decision": "Approved",
        "reimbursement_amount": round(baseline * 0.70, 2),
        "eligible_amount": round(baseline * 0.70, 2),
        "reason": "no_pre_auth_under_threshold",
        "flagged_items": [],
        "policy_type": "standard",
        "confidence": 75,
    }

def llm_validate_fraud(data: dict, baseline: float, fraud_result: dict) -> dict:
    fraud_level = fraud_result.get("label", "LOW")
    if fraud_level != "HIGH_RISK":
        return {
            "decision": "Approved",
            "reimbursement_amount": baseline,
            "eligible_amount": baseline,
            "reason": "rule_based_approved",
            "flagged_items": [],
            "policy_type": "standard",
            "confidence": 85,
        }
    prompt = f'Fraud score: {fraud_result.get("fraud_score", 0):.3f}. Baseline: ${baseline:,.2f}. Fraud? JSON: {{"is_fraud": false}}'
    try:
        res = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            temperature=0.0,
            max_tokens=20,
            timeout=8,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        result = safe_json(res.choices[0].message.content) or {"is_fraud": False}
        if result.get("is_fraud"):
            return {
                "decision": "Denied",
                "reimbursement_amount": 0.0,
                "eligible_amount": 0.0,
                "reason": "fraud_detected",
                "flagged_items": fraud_result.get("flags", []),
                "policy_type": "high_risk",
                "confidence": 92,
            }
        return {
            "decision": "Approved",
            "reimbursement_amount": baseline,
            "eligible_amount": baseline,
            "reason": "fraud_checked_approved",
            "flagged_items": [],
            "policy_type": "standard",
            "confidence": 88,
        }
    except Exception:
        return {
            "decision": "Approved",
            "reimbursement_amount": baseline,
            "eligible_amount": baseline,
            "reason": "fraud_check_error",
            "flagged_items": [],
            "policy_type": "standard",
            "confidence": 80,
        }

def get_decision(query=None, rag_context: str = "", itemized_items: list = None, fraud_result: dict = None, shap_factors: list = None) -> dict:
    raw = query if isinstance(query, dict) else {}
    parsed = parse_query(query)
    claim_amount = _get_amount(parsed, raw)
    if claim_amount <= 0:
        claim_amount = 60.0
    data = _build_data(parsed, claim_amount, raw)
    data["claim_amount"] = max(float(data.get("claim_amount", 0.0)), 1.0)
    reimbursement, breakdown, rule_applied = calculate_dynamic_reimbursement(data["claim_amount"], data)
    llm_result = _fast_decision(data, reimbursement)
    fraud_level = fraud_result.get("label", "LOW") if fraud_result else "LOW"
    if fraud_level == "HIGH_RISK":
        llm_result = llm_validate_fraud(data, reimbursement, fraud_result)
    decision = llm_result.get("decision", "Approved")
    final_amount = float(llm_result.get("reimbursement_amount", reimbursement))
    if data["pre_auth"] == "No" and claim_amount > PRE_AUTH_THRESHOLD:
        final_amount = 0.0
        decision = "Denied"
    elif reimbursement > 0:
        min_allowed = reimbursement * 0.50
        max_allowed = reimbursement * 2.00
        final_amount = max(min_allowed, min(max_allowed, final_amount))
    llm_result["decision"] = decision
    llm_result["reimbursement_amount"] = round(final_amount, 2)
    company_name = breakdown.get("company_name", "Default Insurance")
    recommendation = rag_context.strip() if rag_context else None
    knowledge_reference = f"Additional context from the claim knowledge base suggests: {recommendation}" if recommendation else None
    explanation = _build_human_explanation(
        data,
        breakdown,
        decision,
        final_amount,
        confidence=llm_result.get("confidence", 82),
        shap_factors=shap_factors or [],
        knowledge_reference=knowledge_reference,
    )
    return {
        "status": "ok",
        "decision": decision,
        "reason": llm_result.get("reason", "processed"),
        "reimbursement_amount": round(final_amount, 2),
        "baseline_amount": round(reimbursement, 2),
        "adjustment_percent": round((final_amount / reimbursement - 1) * 100, 1) if reimbursement > 0 else 0,
        "explanation": explanation,
        "flagged_items": llm_result.get("flagged_items", []),
        "policy_type": llm_result.get("policy_type", "standard"),
        "confidence": llm_result.get("confidence", 82),
        "policy_number": data["policy_number"],
        "date_of_service": data["date_of_service"],
        "hospital_name": data["hospital_name"],
        "pre_auth": data["pre_auth"],
        "insurance_company": company_name,
        "rule_applied": rule_applied,
        "reimbursement_percent": breakdown.get("reimbursement_percent", 0.80),
    }

def _save_claim(db_claims, data: dict, decision_result: dict, raw_ocr_text: str = "") -> bool:
    if not db_claims:
        return False
    try:
        from ..models.models_user import Claim
        claim = Claim(
            claim_id=str(uuid.uuid4()),
            patient_age=data.get("age"),
            patient_gender=data.get("gender"),
            patient_income=data.get("patient_income"),
            patient_employment_status=data.get("patient_employment"),
            patient_marital_status=data.get("patient_marital"),
            policy_number=data.get("policy_number"),
            date_of_service=data.get("date_of_service"),
            hospital_name=data.get("hospital_name"),
            pre_authorization_status=data.get("pre_auth"),
            claim_type=data.get("claim_type"),
            diagnosis_code=data.get("diagnosis_code"),
            procedure_code=data.get("procedure_code"),
            claim_amount=data.get("claim_amount"),
            claim_status=decision_result.get("claim_status", "Approved"),
            reimbursement_amount=decision_result.get("reimbursement_amount", 0.0),
            decision=decision_result.get("decision"),
            confidence=decision_result.get("confidence"),
            insurance_company=decision_result.get("insurance_company", ""),
            rule_applied=decision_result.get("rule_applied", "default"),
            raw_ocr_text=raw_ocr_text[:10000] if raw_ocr_text else None,
        )
        db_claims.add(claim)
        db_claims.commit()
        return True
    except Exception:
        db_claims.rollback()
        return False

def process_claim(file_bytes: bytes = None, form_data: dict = None, db_knowledge=None, db_claims=None) -> dict:
    raw_text = ""
    if file_bytes:
        from .ocr_service import ocr_image, clean_text
        raw_text = clean_text(ocr_image(file_bytes))
    extracted_data = extract_claim_data(raw_text) if raw_text else {}
    merged_data = form_data if form_data else {}
    if extracted_data:
        for k, v in extracted_data.items():
            if k not in merged_data and v and str(v).strip():
                merged_data[k] = v
    parsed = parse_query(merged_data)
    claim_amount = _get_amount(parsed, merged_data)
    data = _build_data(parsed, claim_amount, merged_data)
    diagnosis_code = data.get("diagnosis_code") or merged_data.get("DiagnosisCode") or merged_data.get("diagnosis_code") or "Z00.00"
    items = []
    saved_count = 0
    if raw_text:
        items = extract_itemized_bill(raw_text, diagnosis_code)
        if db_knowledge and items:
            for item in items:
                _, action = upsert_knowledge_item(db_knowledge, item, diagnosis_code, "ocr_extracted")
                if action in ("created", "updated"):
                    saved_count += 1
    similar = vector_search(db_knowledge, str(items or merged_data), diagnosis_code) if db_knowledge and items else []
    rag_context = build_rag_context(items, similar)
    fraud_result = predict_fraud(data)
    shap_contributions = explain_decision(data, fraud_result)
    decision_result = get_decision(query=merged_data, rag_context=rag_context, itemized_items=items, fraud_result=fraud_result, shap_factors=shap_contributions)
    _save_claim(db_claims, data, decision_result, raw_text)
    return {
        "status": "ok",
        "raw_text_preview": raw_text[:2000] if raw_text else None,
        "extracted_data": extracted_data,
        "merged_data": merged_data,
        "itemized_items": items,
        "items_count": len(items),
        "saved_to_knowledge": saved_count,
        "diagnosis_code": diagnosis_code,
        "rag_references_used": len(similar),
        "fraud_score": fraud_result.get("fraud_score", 0),
        "fraud_label": fraud_result.get("label", ""),
        "fraud_flags": fraud_result.get("flags", []),
        "shap_factors": shap_contributions,
        **decision_result,
    }