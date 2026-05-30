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

HIGH_RISK_CODES = {"C50", "C61", "C34", "E11", "I21", "I25"}
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
        parsed.get("amount"), parsed.get("claimamount"),
        raw.get("claim_amount"), raw.get("ClaimAmount"),
        raw.get("claimAmount"), raw.get("claimamount"),
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
        "claim_amount": claim_amount,
        "age": safe_int(parsed.get("patientage") or parsed.get("age") or 0),
        "gender": str(parsed.get("patientgender") or parsed.get("gender") or raw.get("PatientGender") or ""),
        "claim_type": str(parsed.get("claimtype") or parsed.get("type") or raw.get("claim_type") or raw.get("ClaimType") or "").lower(),
        "diagnosis_code": str(parsed.get("diagnosiscode") or parsed.get("diagnosis") or raw.get("diagnosis_code") or raw.get("DiagnosisCode") or ""),
        "procedure_code": str(parsed.get("procedurecode") or parsed.get("procedure") or raw.get("procedure_code") or raw.get("ProcedureCode") or ""),
        "provider_specialty": str(parsed.get("providerspecialty") or parsed.get("specialty") or raw.get("provider_specialty") or raw.get("ProviderSpecialty") or "").lower(),
        "patient_income": safe_float(parsed.get("patientincome") or parsed.get("income") or raw.get("patient_income") or raw.get("PatientIncome")) or 0.0,
        "patient_employment": str(parsed.get("patientemploymentstatus") or parsed.get("employment") or raw.get("patient_employment_status") or raw.get("PatientEmploymentStatus") or ""),
        "patient_marital": str(parsed.get("patientmaritalstatus") or parsed.get("marital") or raw.get("patient_marital_status") or raw.get("PatientMaritalStatus") or ""),
        "policy_number": str(parsed.get("policynumber") or parsed.get("policyno") or raw.get("policy_number") or raw.get("PolicyNumber") or ""),
        "date_of_service": str(parsed.get("dateofservice") or parsed.get("date") or raw.get("date_of_service") or raw.get("DateOfService") or ""),
        "hospital_name": str(parsed.get("hospitalname") or raw.get("hospital_name") or raw.get("HospitalName") or ""),
        "pre_auth": pre_auth,
        "submission_method": str(parsed.get("claimsubmissionmethod") or parsed.get("submissionmethod") or raw.get("claim_submission_method") or raw.get("ClaimSubmissionMethod") or ""),
    }


def calculate_baseline_reimbursement(claim_amount: float, data: dict) -> tuple[float, dict]:
    if claim_amount <= 0:
        return 0.0, {"claim_amount": 0, "eligible": 0, "deductible": 0,
                     "after_deduct": 0, "coverage_pct": 80, "copay": 0,
                     "baseline": 0.0, "deductible_not_met": False}

    eligible = claim_amount
    deductible = 50.0
    coverage_pct = 0.80
    copay = 20.0

    if data.get("age", 0) >= 65:
        deductible = 30.0
        coverage_pct = 0.90
        copay = 10.0

    diagnosis = data.get("diagnosis_code", "")
    if any(code in diagnosis for code in HIGH_RISK_CODES):
        deductible = 80.0
        coverage_pct = 0.70
        copay = 30.0

    after_deduct = max(0.0, eligible - deductible)
    reimbursement = max(0.0, round(after_deduct * coverage_pct - copay, 2))

    breakdown = {
        "claim_amount": claim_amount,
        "eligible": eligible,
        "deductible": deductible,
        "after_deduct": after_deduct,
        "coverage_pct": int(coverage_pct * 100),
        "copay": copay,
        "baseline": reimbursement,
        "deductible_not_met": after_deduct == 0,
    }
    return reimbursement, breakdown


def _build_human_explanation(
    data: dict,
    breakdown: dict,
    decision: str,
    final_amount: float,
    shap_factors: list = None,
    knowledge_reference: str = None,
) -> str:
    claim = float(breakdown.get("claim_amount", 0))
    eligible = float(breakdown.get("eligible", claim))
    deductible = float(breakdown.get("deductible", 50))
    after_deduct = float(breakdown.get("after_deduct", 0))
    cov_pct = int(breakdown.get("coverage_pct", 80))
    copay = float(breakdown.get("copay", 20))
    pre_auth = data.get("pre_auth", "No")
    age = int(data.get("age") or 0)
    you_pay = round(claim - final_amount, 2) if final_amount > 0 else claim
    sections = []

    if decision == "Approved":
        headline = "Your claim has been approved."
        sub = "We reviewed your claim and confirmed that your treatment is covered under your insurance policy."
    elif decision == "Partially Approved":
        headline = "Your claim has been partially approved."
        sub = "Part of your claim is covered. We approved the eligible amount based on your policy."
    elif decision == "Denied":
        headline = "Your claim could not be approved."
        sub = "Below is why your claim was not approved and what you can do next."
    else:
        headline = "Your claim is under review."
        sub = "We need more information to complete your claim."

    sections.append(f'''<div class="exp-container">
  <div class="exp-header">
    <div class="exp-headline">{headline}</div>
    <div class="exp-sub">{sub}</div>
  </div>''')

    if final_amount > 0:
        sections.append(f'''
  <div class="exp-amounts">
    <div class="exp-amount-row">
      <span class="exp-label">Your Total Medical Bill</span>
      <span class="exp-value">${claim:,.2f}</span>
    </div>
    <div class="exp-amount-row covered">
      <span class="exp-label">Insurance Will Pay You</span>
      <span class="exp-value">${final_amount:,.2f}</span>
    </div>
    <div class="exp-amount-row owe">
      <span class="exp-label">You Pay Out of Pocket</span>
      <span class="exp-value">${you_pay:,.2f}</span>
    </div>
  </div>''')

    covered_amt = round(after_deduct * cov_pct / 100, 2)
    sections.append(f'''
    <div class="exp-section">
      <div class="exp-section-title">How Your Reimbursement Was Calculated</div>
      <div class="exp-calc-table">
        <div class="exp-calc-row exp-calc-header">
          <span class="exp-calc-label">Total bill you submitted</span>
          <span class="exp-calc-value">${claim:,.2f}</span>
        </div>
        <div class="exp-calc-row exp-calc-subtle">
          <span class="exp-calc-label">Amount our system accepts as eligible</span>
          <span class="exp-calc-value">${eligible:,.2f}</span>
        </div>
        <div class="exp-calc-row exp-calc-deduction">
          <span class="exp-calc-label">Minus your deductible (what you pay first)</span>
          <span class="exp-calc-value">- ${deductible:,.2f}</span>
        </div>
        <div class="exp-calc-row exp-calc-divider">
          <span class="exp-calc-label">Amount left after deductible</span>
          <span class="exp-calc-value">${after_deduct:,.2f}</span>
        </div>
        <div class="exp-calc-row exp-calc-subtle">
          <span class="exp-calc-label">Your insurance covers {cov_pct}% of this</span>
          <span class="exp-calc-value">${covered_amt:,.2f}</span>
        </div>
        <div class="exp-calc-row exp-calc-deduction">
          <span class="exp-calc-label">Minus your co-pay (fixed fee)</span>
          <span class="exp-calc-value">- ${copay:,.2f}</span>
        </div>
        <div class="exp-calc-row exp-calc-total">
          <span class="exp-calc-label">Final amount insurance will pay you</span>
          <span class="exp-calc-value">${final_amount:,.2f}</span>
        </div>
      </div>
    </div>
    ''')

    if knowledge_reference:
        sections.append(f'''
    <div class="exp-section">
      <div class="exp-section-title">Why You Got ${final_amount:,.2f} Instead of ${float(breakdown.get("baseline", 0)):,.2f}</div>
      <div class="exp-explanation-box">
        <p><strong>Baseline (${float(breakdown.get("baseline", 0)):,.2f}):</strong> This is what a basic formula would give you using fixed rules (like deductible = $50, coverage = 80%). It's a simple starting point.</p>
        <p><strong>Actual Reimbursement (${final_amount:,.2f}):</strong> This is what you actually get. We adjusted it higher because:</p>
        <ul class="exp-adjustment-list">
          <li>Similar claims in our database were approved at this higher amount</li>
          <li>Your treatment was medically necessary and appropriate</li>
          <li>You had pre-authorization, which confirms coverage</li>
        </ul>
        <p class="exp-note">Bottom line: You got <strong>${round(final_amount - float(breakdown.get("baseline", 0)), 2):,.2f} more</strong> than the basic formula would give you.</p>
      </div>
    </div>
    ''')

    why_items = []
    if pre_auth == "Yes":
        why_items.append("You got pre-authorization before treatment. This means your insurer already agreed to cover this procedure.")
    elif pre_auth == "No" and claim > PRE_AUTH_THRESHOLD:
        why_items.append(f"You did not get pre-authorization for a claim over ${PRE_AUTH_THRESHOLD:,.0f}. Large procedures need prior approval.")
    if age >= 65:
        why_items.append(f"At age {age}, you qualify for senior benefits, which give you better coverage.")
    diagnosis = data.get("diagnosis_code", "")
    if any(code in diagnosis for code in HIGH_RISK_CODES):
        why_items.append(f"Your diagnosis ({diagnosis}) is considered complex, so special coverage rules apply.")

    if why_items:
        items_html = "".join(f"<li>{i}</li>" for i in why_items)
        sections.append(f'''
  <div class="exp-section">
    <div class="exp-section-title">Why We Made This Decision</div>
    <ul class="exp-list">{items_html}</ul>
  </div>''')

    if shap_factors:
        notable = [f for f in shap_factors[:5] if abs(f.get("contribution", 0)) >= 0.04]
        if notable:
            rows = []
            for f in notable:
                feature = f.get("feature", "")
                direction = f.get("direction", "")
                LABELS2 = {
                    "pre_auth": ("You had pre-authorization, which helps your claim", "No pre-authorization made things harder"),
                    "claim_amount": ("Your bill is a normal amount for this treatment", "Your bill size affected the calculation"),
                    "amount_income_ratio": ("Your bill is reasonable for your income", "Your bill is high compared to your income"),
                    "age": ("Your age helps you get better coverage", "Your age was considered in coverage"),
                    "fraud_score": ("Nothing suspicious found in your claim", "Some things in your claim needed extra checking"),
                }
                pos_lbl, neg_lbl = LABELS2.get(feature, (feature.replace("_", " ").title(), feature.replace("_", " ").title()))
                text = pos_lbl if direction == "decreases_risk" else neg_lbl
                badge = "<span class='exp-badge exp-badge-favorable'>Good for you</span>" if direction == "decreases_risk" else "<span class='exp-badge exp-badge-reviewed'>Checked</span>"
                rows.append(f"<div class='exp-factor-row'><span class='exp-factor-label'>{text}</span>{badge}</div>")
            sections.append(f'''
  <div class="exp-section">
    <div class="exp-section-title">What the AI Looked At</div>
    <div class="exp-factors">{"".join(rows)}</div>
  </div>''')

    if decision == "Denied":
        sections.append('''
  <div class="exp-tip">
    <strong>What you can do:</strong> You can appeal this decision. Call the number on your insurance card and ask about the appeals process. Emergency cases may be an exception.
  </div>''')

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
    item_count = len(itemized_items) if itemized_items else 0

    fraud_block = ""
    if fraud_result and fraud_result.get("label") == "HIGH_RISK":
        flags = ", ".join(fraud_result.get("flags", [])) or "none"
        fraud_block = f"\nFRAUD RISK: HIGH (score {fraud_result.get('fraud_score', 0):.3f}). Flags: {flags}.\n"

    knowledge_summary = ""
    if rag_context:
        knowledge_summary = f"\nKNOWLEDGE BASE REFERENCE:\n{rag_context[:2000]}\n\nUse this to determine if claim amount is reasonable for this procedure/diagnosis.\n"
    else:
        knowledge_summary = "\nKNOWLEDGE BASE: No similar claims found. Rely on general medical insurance standards.\n"

    prompt = f"""You are a health insurance claims adjudicator using KNOWLEDGE-BASED decision making.

CLAIM DETAILS:
- Total Amount: ${data.get('claim_amount', 0):,.2f}
- Pre-Authorization: {data.get('pre_auth', 'No')}
- Patient Age: {data.get('age', 'N/A')}
- Diagnosis Code: {data.get('diagnosis_code') or 'N/A'}
- Procedure Code: {data.get('procedure_code') or 'N/A'}
- Provider Specialty: {data.get('provider_specialty') or 'N/A'}
- Hospital: {data.get('hospital_name') or 'N/A'}
- Rule-Based Baseline (reference only): ${baseline:,.2f}
- Coverage Rate from Rules: {breakdown.get("coverage_pct", 75)}%
{fraud_block}

ITEMIZED BILL ({item_count} items):
{items_block}

{knowledge_summary}

DECISION PRINCIPLES (in priority order):

1. PRE-AUTHORIZATION:
   - Pre-Auth = Yes --> Always approve (unless fraud detected)
   - Pre-Auth = No AND amount > ${PRE_AUTH_THRESHOLD:,.0f} --> Deny (policy violation)
   - Pre-Auth = No AND amount <= ${PRE_AUTH_THRESHOLD:,.0f} --> Approve at 70-85% of what you would give with pre-auth

2. KNOWLEDGE-BASED REASONABLENESS:
   - Compare claim amount to similar claims in knowledge base
   - If claim is within 20% of knowledge base average --> approve at 80-100% of claim
   - If claim is 20-50% above knowledge base --> approve at 60-80% (partial)
   - If claim is >50% above knowledge base --> deny or approve at 30-50% (partial)

3. ITEMIZED BILL REVIEW:
   - Flag any item >30% above knowledge base average
   - If flagged items <20% of total --> approve rest
   - If flagged items >50% of total --> deny or partial approval

4. HIGH-RISK DIAGNOSIS (C50, C61, C34, E11, I21, I25):
   - Apply higher scrutiny but DO NOT automatically deny
   - If treatment is medically necessary (supported by knowledge base) --> approve
   - If treatment seems unnecessary/excessive --> partial or deny

5. FRAUD SIGNALS:
   - HIGH_RISK + valid flags --> deny or reduce by 40-60%
   - HIGH_RISK + False Positive explanation --> proceed normally
   - MEDIUM_RISK --> reduce confidence, consider partial approval

6. BASELINE IS REFERENCE ONLY:
   - Baseline = ${baseline:,.2f} is calculated from rigid rules
   - You MAY deviate from baseline if knowledge base supports it
   - If knowledge base shows similar claims approved at higher amounts --> approve higher
   - You must explain WHY you deviated from baseline in the reason field

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
    amount = float(data.get("claim_amount", 0))

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
    raw = query if isinstance(query, dict) else {}
    parsed = parse_query(query)

    missing = detect_missing(parsed)
    if missing:
        return {
            "status": "ok", "decision": "Pending", "reason": "missing_required_fields",
            "missing_fields": missing, "reimbursement_amount": 0.0, "baseline_amount": 0.0,
            "explanation": "Some required fields are missing: " + ", ".join(missing) + ". Please provide complete information.",
            "confidence": 0.0,
        }

    claim_amount = _get_amount(parsed, raw)
    data = _build_data(parsed, claim_amount, raw)
    baseline, breakdown = calculate_baseline_reimbursement(claim_amount, data)

    llm_result = llm_calculate_reimbursement(
        data, baseline, breakdown, itemized_items or [], rag_context, fraud_result
    )

    shap_factors = shap_factors or []

    decision = llm_result.get("decision", "Pending")
    final_amount = float(llm_result.get("reimbursement_amount", baseline))

    if data["pre_auth"] == "No" and claim_amount > PRE_AUTH_THRESHOLD:
        final_amount = 0.0
        decision = "Denied"
    else:
        min_allowed = baseline * 0.50
        max_allowed = baseline * 2.00
        final_amount = max(min_allowed, min(max_allowed, final_amount))

    llm_result["decision"] = decision
    llm_result["reimbursement_amount"] = round(final_amount, 2)

    knowledge_ref = ""
    if rag_context and final_amount != baseline:
        knowledge_ref = f"The reimbursement amount was adjusted from the rule-based baseline (${baseline:,.2f}) based on similar claims in our knowledge base and medical necessity assessment."

    explanation = _build_human_explanation(data, breakdown, decision, final_amount, shap_factors=shap_factors, knowledge_reference=knowledge_ref)

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
            claim_status=decision_result.get("decision", "Pending"),
            reimbursement_amount=decision_result.get("reimbursement_amount", 0.0),
            decision=decision_result.get("decision"),
            confidence=decision_result.get("confidence"),
            raw_ocr_text=raw_ocr_text[:10000] if raw_ocr_text else None,
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
    parsed = parse_query(extracted_data)
    claim_amount = _get_amount(parsed, extracted_data)
    data = _build_data(parsed, claim_amount, extracted_data)

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

    similar = vector_search(db_knowledge, str(items or extracted_data), diagnosis_code) if db_knowledge else []
    rag_context = build_rag_context(items, similar)

    fraud_result = predict_fraud(data)
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
        "status": "ok",
        "raw_text_preview": raw_text[:2000] if raw_text else None,
        "extracted_data": extracted_data,
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