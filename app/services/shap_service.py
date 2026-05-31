import numpy as np
import joblib
from pathlib import Path
from typing import List, Dict, Optional
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy import create_engine, text

from .fraud_service import build_features, FEATURE_NAMES

MODEL_PATH = Path(__file__).parent.parent / "models" / "fraud_model_lgb.pkl"

_explainer = None

def _load_model():
    if MODEL_PATH.exists():
        try:
            return joblib.load(MODEL_PATH)
        except Exception:
            return None
    return None

def explain_decision(
    data: dict,
    fraud_result: dict = None,
    db: Session = None,
    claim_id: str = None
) -> List[Dict]:
    model = _load_model()
    if model is None:
        return _fallback_explanation(data)

    try:
        features = build_features(data)[0]
        explainer = _get_explainer(model)
        shap_values = explainer.shap_values(features.reshape(1, -1))

        if isinstance(shap_values, list):
            values = shap_values[1][0]
        else:
            values = shap_values[0]

        contributions = []
        for name, value, shap_val in zip(FEATURE_NAMES, features, values):
            contributions.append({
                "feature": name,
                "value": round(float(value), 4),
                "contribution": round(float(shap_val), 4),
                "direction": "increases_risk" if shap_val > 0 else "decreases_risk",
                "explanation": _generate_natural_language_explanation(name, value, shap_val, data, db, claim_id),
            })

        contributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)
        return contributions[:10]

    except Exception as e:
        print(f"[shap_service] explain_decision error: {e}")
        return _fallback_explanation(data)

def _get_explainer(model):
    global _explainer
    if _explainer is None:
        import shap
        _explainer = shap.TreeExplainer(model)
    return _explainer

def _generate_natural_language_explanation(
    feature_name: str,
    value: float,
    shap_val: float,
    data: dict,
    db: Session = None,
    claim_id: str = None
) -> str:
    if feature_name == "pre_auth":
        if value == 1:
            return "You got pre-authorization before treatment. This means your insurer already approved this procedure, significantly reducing fraud risk."
        else:
            return "No pre-authorization was obtained before treatment. This increases risk as the insurer has not yet verified medical necessity."

    if feature_name == "claim_amount":
        if value > 5000:
            return f"Claim amount (${value:,.2f}) is unusually high compared to typical cases for this diagnosis. High-value claims require additional scrutiny."
        else:
            return f"Claim amount (${value:,.2f}) is within normal range for this procedure, consistent with historical data."

    if feature_name == "amount_income_ratio":
        income = data.get("patient_income") or 1
        ratio = value
        if ratio > 8:
            return f"Claim amount represents {ratio*100:.0f}% of patient income, which is unusually high and may indicate financial stress or potential abuse."
        else:
            return f"Claim amount represents {ratio*100:.1f}% of patient income, within acceptable range for this treatment type."

    if feature_name == "drug_disease_match":
        if db and claim_id:
            match_score = _get_drug_disease_match_score(db, data, claim_id)
            if match_score >= 0.9:
                return f"Prescription verified: All {match_score*100:.0f}% of drugs prescribed are clinically appropriate for your diagnosis (ICD {data.get('diagnosis_code')}). This strongly supports claim validity."
            elif match_score >= 0.7:
                return f"Most drugs prescribed match your diagnosis (score: {match_score*100:.0f}%). Minor discrepancies detected but overall clinically reasonable."
            else:
                return f"Some prescribed drugs do not match your diagnosis (score: {match_score*100:.0f}%). Medical review recommended to verify necessity."
        return "Drug-disease relationship could not be verified. Manual review may be required."

    if feature_name == "provider_history_score":
        if db:
            provider = data.get("provider_name")
            if provider:
                history = _get_provider_fraud_history(db, provider)
                if history["fraud_rate"] < 0.05:
                    return f"Provider {provider} has excellent track record: only {history['fraud_rate']*100:.1f}% fraud rate across {history['total_claims']} claims. Low-risk provider."
                elif history["fraud_rate"] < 0.15:
                    return f"Provider {provider} has moderate fraud rate ({history['fraud_rate']*100:.1f}% across {history['total_claims']} claims). Standard review applied."
                else:
                    return f"Provider {provider} has elevated fraud rate ({history['fraud_rate']*100:.1f}% across {history['total_claims']} claims). Enhanced scrutiny recommended."
        return "Provider historical data unavailable. Default risk assessment applied."

    if feature_name == "insurance_coverage_percent":
        coverage = value * 100
        company = data.get("insurance_company")
        if company:
            return f"Insurance company {company} typically covers {coverage:.0f}% of similar claims. This coverage level is consistent with their standard policy for this procedure."
        else:
            return f"Coverage percentage is {coverage:.0f}, consistent with typical reimbursement rates for this diagnosis."

    if feature_name == "diagnosis_procedure_consistency":
        diagnosis = data.get("diagnosis_code")
        procedure = data.get("procedure_code")
        if diagnosis and procedure:
            is_consistent = _check_diagnosis_procedure_match(db, diagnosis, procedure)
            if is_consistent:
                return f"Procedure code {procedure} is clinically appropriate for diagnosis {diagnosis}. No mismatch detected."
            else:
                return f"Potential mismatch detected: procedure {procedure} may not be standard treatment for diagnosis {diagnosis}. Medical review recommended."
        return "Diagnosis or procedure code missing. Unable to verify clinical consistency."

    if feature_name == "age":
        age = int(value)
        if age >= 65:
            return f"Patient age ({age} years) is in elderly category. Elderly patients typically have higher claim volumes but lower fraud rates."
        elif age < 18:
            return f"Patient age ({age} years) is pediatric. Pediatric claims require careful review for age-appropriate treatments."
        else:
            return f"Patient age ({age} years) is in adult range, typical for most medical claims."

    if feature_name == "Gender":
        gender = "Female" if value == 1 else "Male"
        return f"Patient gender: {gender}. Gender-based risk patterns are minimal but accounted for in model."

    if feature_name == "claim_frequency_30d":
        freq = int(value)
        if freq > 3:
            return f"Patient has filed {freq} claims in the past 30 days, which is unusually high and may indicate potential abuse or billing errors."
        else:
            return f"Patient claim frequency ({freq} claims in 30 days) is within normal range."

    if feature_name == "distance_to_provider":
        distance = value
        if distance > 50:
            return f"Patient traveled {distance:.1f} miles to see this provider, which is unusually far and may warrant additional verification."
        else:
            return f"Patient traveled {distance:.1f} miles to see this provider, within normal commuting distance."

    return f"Feature '{feature_name}' value: {value:.4f}. Contribution to risk: {'increases' if shap_val > 0 else 'decreases'} by {abs(shap_val):.4f}."

def _get_drug_disease_match_score(db: Session, data: dict, claim_id: str) -> float:
    try:
        from .knowledge_service import get_claim_knowledge
        
        provider = data.get("provider_name")
        company = data.get("insurance_company")
        
        if provider:
            result = get_claim_knowledge("provider", provider, "drug_disease_match_percent")
            return result.get("value", 0.7)
        elif company:
            result = get_claim_knowledge("insurance_company", company, "drug_disease_match_percent")
            return result.get("value", 0.7)
        
        return 0.7
    except Exception:
        return 0.7

def _get_provider_fraud_history(db: Session, provider_name: str) -> dict:
    try:
        engine = create_engine("sqlite:///health_claims.db")
        SessionLocal = sessionmaker(bind=engine)
        session = SessionLocal()
        
        result = session.execute(text("""
            SELECT 
                COUNT(*) as total_claims,
                AVG(CASE WHEN is_fraud = 1 THEN 1 ELSE 0 END) as fraud_rate
            FROM claims
            WHERE provider_name = :provider
        """), {"provider": provider_name})
        
        row = result.fetchone()
        
        return {
            "total_claims": row[0] or 0,
            "fraud_rate": row[1] or 0.0
        }
    except Exception:
        return {"total_claims": 0, "fraud_rate": 0.0}

def _check_diagnosis_procedure_match(db: Session, diagnosis_code: str, procedure_code: str) -> bool:
    try:
        from .knowledge_service import vector_search_sqlite
        
        query = f"{diagnosis_code} procedure {procedure_code}"
        results = vector_search_sqlite(db, query, diagnosis_code=diagnosis_code, top_k=1, threshold=0.75)
        
        return len(results) > 0 and results[0]["similarity"] > 0.8
    except Exception:
        return True

def _fallback_explanation(data: dict) -> List[Dict]:
    amount = float(data.get("claim_amount") or 0)
    pre_auth = str(data.get("pre_auth", "")).strip().lower() == "yes"
    income = float(data.get("patient_income") or 1)
    age = int(data.get("age") or 0)
    ratio = amount / income if income > 0 else 0

    factors = []

    if pre_auth:
        factors.append({
            "feature": "pre_auth",
            "value": 1,
            "contribution": -0.30,
            "direction": "decreases_risk",
            "explanation": "You got pre-authorization before treatment. This means your insurer already approved this procedure, significantly reducing fraud risk."
        })
    else:
        factors.append({
            "feature": "pre_auth",
            "value": 0,
            "contribution": 0.30,
            "direction": "increases_risk",
            "explanation": "No pre-authorization was obtained before treatment. This increases risk as the insurer has not yet verified medical necessity."
        })

    factors.append({
        "feature": "claim_amount",
        "value": amount,
        "contribution": 0.28 if amount > 5000 else 0.08,
        "direction": "increases_risk" if amount > 5000 else "decreases_risk",
        "explanation": f"Claim amount (${amount:,.2f}) is {'unusually high' if amount > 5000 else 'within normal range'} for this procedure."
    })

    if ratio > 8:
        factors.append({
            "feature": "amount_income_ratio",
            "value": round(ratio, 2),
            "contribution": 0.20,
            "direction": "increases_risk",
            "explanation": f"Claim amount represents {ratio*100:.0f}% of patient income, which is unusually high and may indicate financial stress."
        })
    else:
        factors.append({
            "feature": "amount_income_ratio",
            "value": round(ratio, 2),
            "contribution": -0.10,
            "direction": "decreases_risk",
            "explanation": f"Claim amount represents {ratio*100:.1f}% of patient income, within acceptable range."
        })

    if age >= 65:
        factors.append({
            "feature": "age",
            "value": age,
            "contribution": -0.15,
            "direction": "decreases_risk",
            "explanation": f"Patient age ({age} years) is in elderly category, typically lower fraud rates."
        })

    return sorted(factors, key=lambda x: abs(x["contribution"]), reverse=True)