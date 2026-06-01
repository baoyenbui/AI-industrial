import uuid
import json
from groq import Groq
import os
from dotenv import load_dotenv
import sqlite3
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
REQUIRED_FORM_FIELDS = ["PatientAge", "PatientGender", "PatientEmploymentStatus", "ClaimAmount", "DiagnosisCode", "ProviderName", "PreAuthorizationStatus", "ProcedureCode", "ClaimSubmissionMethod", "ProviderSpecialty", "ClaimType", "PatientMaritalStatus", "PolicyNumber"]


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
    reimbursement = max(0.0, round(after_deduct * applied_percent, 2))

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
        "baseline": reimbursement,
        "deductible_not_met": after_deduct == 0,
        "reimbursement_percent": applied_percent,
        "rule_applied": rule_applied,
        "rule_label": rule_label,
        "max_amount": max_amount,
        "company_id": company_rules["company_id"],
        "company_name": company_name,
    }
    return reimbursement, breakdown, rule_applied


def _why_items(data: dict, breakdown: dict, decision: str, final_amount: float, shap_factors: list = None, knowledge_reference: str = None):
    claim = float(breakdown.get("claim_amount", 60))
    deductible = float(breakdown.get("deductible", 50))
    after_deduct = float(breakdown.get("after_deduct", 0))
    cov_pct = int(breakdown.get("coverage_pct", 80))
    pre_auth = data.get("pre_auth", "No")
    age = int(data.get("age") or 30)
    diagnosis = data.get("diagnosis_code", "Z00.00")
    procedure_code = data.get("procedure_code", "99213")
    provider = data.get("hospital_name", "Unknown Provider")
    specialty = data.get("provider_specialty", "general")
    claim_type = data.get("claim_type", "medical")
    company_name = breakdown.get("company_name", "Default Insurance")
    rule_label = breakdown.get("rule_label", breakdown.get("rule_applied", "standard policy"))

    items = []
    if decision == "Approved":
        items.append("Your claim was approved because it fits the insurer’s coverage pattern for this type of treatment.")
    elif decision == "Denied":
        items.append("Your claim could not be approved because it did not satisfy the policy conditions required for payment.")
    else:
        items.append("Your claim is still under review because the available information is not enough for a final decision.")

    if pre_auth == "Yes":
        items.append("Pre-authorization was confirmed before treatment, which supports a smoother approval path.")
    else:
        items.append("No pre-authorization was confirmed before treatment, so the claim was evaluated under the standard policy path.")

    if any(code in diagnosis for code in HIGH_RISK_CODES):
        items.append(f"The diagnosis code {diagnosis} belongs to a higher-complexity category, which can affect the deductible and review level.")
    else:
        items.append(f"The diagnosis code {diagnosis} was handled under the standard clinical profile for this claim type.")

    items.append(f"This is a {claim_type} claim, and the procedure code {procedure_code} is consistent with the submitted treatment type.")

    if provider and provider != "Unknown Provider":
        items.append(f"The provider {provider} was considered together with historical claim patterns and policy context from the knowledge base.")
    else:
        items.append("The provider was not clearly identified, so the system relied more heavily on the claim details and policy rules.")

    if specialty and specialty != "general":
        items.append(f"The provider specialty is {specialty}, which usually influences how similar claims are reimbursed.")

    if age >= 65:
        items.append(f"At age {age}, the claim may receive different handling than a younger patient claim under the policy rules.")

    if shap_factors:
        top = []
        for x in shap_factors[:3]:
            feature = str(x.get("feature", "unknown")).replace("_", " ")
            expl = str(x.get("explanation", "")).strip()
            if expl:
                top.append(f"{feature}: {expl}")
        if top:
            items.append("The strongest model signals were: " + "; ".join(top) + ".")

    if knowledge_reference:
        items.append(knowledge_reference)

    if company_name:
        items.append(f"This claim was evaluated under {company_name}'s {rule_label} coverage path.")

    return items, claim, deductible, after_deduct, cov_pct


def _format_bullets(items):
    return "".join(f"<li>{x}</li>" for x in items if x)


def _build_human_explanation(data: dict, breakdown: dict, decision: str, final_amount: float, shap_factors: list = None, knowledge_reference: str = None) -> str:
    items, claim, deductible, after_deduct, cov_pct = _why_items(data, breakdown, decision, final_amount, shap_factors, knowledge_reference)
    amount_payable = round(final_amount, 2)
    out_of_pocket = round(max(claim - amount_payable, 0.0), 2)

    if decision == "Approved":
        headline = "Your claim has been approved."
        sub = "Your treatment is covered under your insurance policy."
    elif decision == "Denied":
        headline = "Your claim could not be approved."
        sub = "The available claim information did not satisfy the policy conditions."
    else:
        headline = "Your claim is under review."
        sub = "We need a little more information before a final decision can be made."

    return f'''<div class="exp-container">
  <div class="exp-header">
    <div class="exp-headline">{headline}</div>
    <div class="exp-sub">{sub}</div>
  </div>
  <div class="exp-amounts">
    <div class="exp-amount-row">
      <span class="exp-label">Your Total Medical Bill</span>
      <span class="exp-value">${claim:,.2f}</span>
    </div>
    <div class="exp-amount-row covered">
      <span class="exp-label">Insurance Will Pay You</span>
      <span class="exp-value">${amount_payable:,.2f}</span>
    </div>
    <div class="exp-amount-row owe">
      <span class="exp-label">You Pay Out of Pocket</span>
      <span class="exp-value">${out_of_pocket:,.2f}</span>
    </div>
  </div>
  <div class="exp-section">
    <div class="exp-section-title">How Your Reimbursement Was Calculated</div>
    <div class="exp-calc-table">
      <div class="exp-calc-row exp-calc-header"><span class="exp-calc-label">Total bill</span><span class="exp-calc-value">${claim:,.2f}</span></div>
      <div class="exp-calc-row exp-calc-deduction"><span class="exp-calc-label">Minus deductible</span><span class="exp-calc-value">- ${deductible:,.2f}</span></div>
      <div class="exp-calc-row exp-calc-subtle"><span class="exp-calc-label">After deductible</span><span class="exp-calc-value">${after_deduct:,.2f}</span></div>
      <div class="exp-calc-row exp-calc-subtle"><span class="exp-calc-label">Coverage ({cov_pct}%)</span><span class="exp-calc-value">${round(after_deduct * cov_pct / 100, 2):,.2f}</span></div>
      <div class="exp-calc-row exp-calc-total"><span class="exp-calc-label">Final reimbursement</span><span class="exp-calc-value">${amount_payable:,.2f}</span></div>
    </div>
  </div>
  <div class="exp-section">
    <div class="exp-section-title">Why We Made This Decision</div>
    <ul class="exp-list">{_format_bullets(items)}</ul>
  </div>
</div>'''


def _fast_decision(data: dict, baseline: float) -> dict:
    pre_auth = data.get("pre_auth", "No")
    amount = float(data.get("claim_amount", 60))
    if pre_auth == "Yes":
        return {"decision": "Approved", "reimbursement_amount": round(baseline * 0.85, 2), "eligible_amount": round(baseline * 0.85, 2), "reason": "pre_authorization_confirmed", "flagged_items": [], "policy_type": "standard", "confidence": 0.82}
    if amount > PRE_AUTH_THRESHOLD:
        return {"decision": "Denied", "reimbursement_amount": 0.0, "eligible_amount": 0.0, "reason": "no_pre_auth_exceeds_threshold", "flagged_items": [], "policy_type": "standard", "confidence": 0.90}
    return {"decision": "Approved", "reimbursement_amount": round(baseline * 0.70, 2), "eligible_amount": round(baseline * 0.70, 2), "reason": "no_pre_auth_under_threshold", "flagged_items": [], "policy_type": "standard", "confidence": 0.75}


def llm_validate_fraud(data: dict, baseline: float, fraud_result: dict) -> dict:
    fraud_level = fraud_result.get("label", "LOW")
    if fraud_level != "HIGH_RISK":
        return {"decision": "Approved", "reimbursement_amount": baseline, "eligible_amount": baseline, "reason": "rule_based_approved", "flagged_items": [], "policy_type": "standard", "confidence": 0.85}
    prompt = f"Fraud score: {fraud_result.get('fraud_score', 0):.3f}. Baseline: ${baseline:,.2f}. Fraud? JSON: {{\"is_fraud\": false}}"
    try:
        res = client.chat.completions.create(model="llama-3.3-70b-versatile", temperature=0.0, max_tokens=20, timeout=8, messages=[{"role": "user", "content": prompt}], response_format={"type": "json_object"})
        result = safe_json(res.choices[0].message.content) or {"is_fraud": False}
        if result.get("is_fraud"):
            return {"decision": "Denied", "reimbursement_amount": 0.0, "eligible_amount": 0.0, "reason": "fraud_detected", "flagged_items": fraud_result.get("flags", []), "policy_type": "high_risk", "confidence": 0.92}
        return {"decision": "Approved", "reimbursement_amount": baseline, "eligible_amount": baseline, "reason": "fraud_checked_approved", "flagged_items": [], "policy_type": "standard", "confidence": 0.88}
    except Exception:
        return {"decision": "Approved", "reimbursement_amount": baseline, "eligible_amount": baseline, "reason": "fraud_check_error", "flagged_items": [], "policy_type": "standard", "confidence": 0.80}


def get_decision(query=None, rag_context: str = "", itemized_items: list = None, fraud_result: dict = None, shap_factors: list = None) -> dict:
    raw = query if isinstance(query, dict) else {}
    parsed = parse_query(query)
    claim_amount = _get_amount(parsed, raw)
    if claim_amount <= 0:
        claim_amount = 60.0
    data = _build_data(parsed, claim_amount, raw)
    data["claim_amount"] = max(float(data.get("claim_amount", 0.0)), 1.0)
    baseline, breakdown, rule_applied = calculate_dynamic_reimbursement(data["claim_amount"], data)
    llm_result = _fast_decision(data, baseline)
    fraud_level = fraud_result.get("label", "LOW") if fraud_result else "LOW"
    if fraud_level == "HIGH_RISK":
        llm_result = llm_validate_fraud(data, baseline, fraud_result)
    decision = llm_result.get("decision", "Approved")
    final_amount = float(llm_result.get("reimbursement_amount", baseline))
    if data["pre_auth"] == "No" and claim_amount > PRE_AUTH_THRESHOLD:
        final_amount = 0.0
        decision = "Denied"
    elif baseline > 0:
        min_allowed = baseline * 0.50
        max_allowed = baseline * 2.00
        final_amount = max(min_allowed, min(max_allowed, final_amount))
    llm_result["decision"] = decision
    llm_result["reimbursement_amount"] = round(final_amount, 2)
    company_name = breakdown.get("company_name", "Default Insurance")
    recommendation = rag_context.strip() if rag_context else None
    knowledge_reference = f"Additional context from the claim knowledge base suggests: {recommendation}" if recommendation else None
    explanation = _build_human_explanation(data, breakdown, decision, final_amount, shap_factors=shap_factors or [], knowledge_reference=knowledge_reference)
    return {
        "status": "ok",
        "decision": decision,
        "reason": llm_result.get("reason", "processed"),
        "reimbursement_amount": round(final_amount, 2),
        "baseline_amount": round(baseline, 2),
        "adjustment_percent": round((final_amount / baseline - 1) * 100, 1) if baseline > 0 else 0,
        "explanation": explanation,
        "flagged_items": llm_result.get("flagged_items", []),
        "policy_type": llm_result.get("policy_type", "standard"),
        "confidence": llm_result.get("confidence", 0.75),
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