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


def _get_explainer(model):
    global _explainer
    if _explainer is None:
        import shap
        _explainer = shap.TreeExplainer(model)
    return _explainer


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
            return "Pre-authorization was confirmed before treatment. This usually strengthens the claim because the insurer had already reviewed the service."
        return "No pre-authorization was found before treatment. This often makes the claim look less certain and may require more review."

    if feature_name == "claim_amount":
        if value > 5000:
            return f"The claim amount of ${value:,.2f} is high, so it deserves closer review compared with typical claims."
        return f"The claim amount of ${value:,.2f} is within a common range for this type of claim."

    if feature_name == "amount_income_ratio":
        income = data.get("patient_income") or 1
        ratio = value
        if ratio > 8:
            return f"The claim amount is very large compared with the patient’s income, which can make the claim stand out more strongly."
        return f"The claim amount is reasonably aligned with the patient’s income level."

    if feature_name == "drug_disease_match":
        match_score = _get_drug_disease_match_score(db, data, claim_id)
        diagnosis = data.get("diagnosis_code", "UNKNOWN")
        if match_score >= 0.9:
            return f"The prescription appears highly appropriate for diagnosis {diagnosis}. This strongly supports medical necessity."
        if match_score >= 0.7:
            return f"Most of the prescribed medication appears consistent with diagnosis {diagnosis}, so the treatment looks broadly reasonable."
        return f"The medication and diagnosis match is weaker for diagnosis {diagnosis}, so the claim may need a closer medical review."

    if feature_name == "provider_history_score":
        provider = data.get("provider_name")
        history = _get_provider_fraud_history(db, provider) if provider else {"fraud_rate": 0.0, "total_claims": 0}
        if provider and history["total_claims"] > 0:
            if history["fraud_rate"] < 0.05:
                return f"The provider {provider} has a strong low-risk history across {history['total_claims']} claims."
            if history["fraud_rate"] < 0.15:
                return f"The provider {provider} has a moderate historical risk profile across {history['total_claims']} claims."
            return f"The provider {provider} has a higher historical risk profile across {history['total_claims']} claims."
        return "Provider history is not available, so the model relies more heavily on the current claim details."

    if feature_name == "insurance_coverage_percent":
        coverage = value * 100
        company = data.get("insurance_company")
        if company:
            return f"The insurer {company} typically covers about {coverage:.0f}% of similar claims, which guides the reimbursement level."
        return f"The coverage rate is about {coverage:.0f}%, which directly influences the final reimbursement amount."

    if feature_name == "diagnosis_procedure_consistency":
        diagnosis = data.get("diagnosis_code")
        procedure = data.get("procedure_code")
        if diagnosis and procedure:
            if _check_diagnosis_procedure_match(db, diagnosis, procedure):
                return f"The procedure code {procedure} looks clinically consistent with diagnosis {diagnosis}."
            return f"The procedure code {procedure} may not be the most typical match for diagnosis {diagnosis}, so the claim may deserve more review."
        return "Diagnosis or procedure information is incomplete, so consistency could not be fully verified."

    if feature_name == "age":
        age = int(value)
        if age >= 65:
            return f"The patient is {age}, which places the claim in an older age group that often follows a different reimbursement pattern."
        if age < 18:
            return f"The patient is {age}, which places the claim in a pediatric group that often needs age-specific handling."
        return f"The patient is {age}, which is a standard adult profile for most claims."

    if feature_name == "Gender":
        gender = "Female" if value == 1 else "Male"
        return f"The patient is recorded as {gender}, which is included in the model but usually has a smaller effect than claim amount or pre-authorization."

    if feature_name == "claim_frequency_30d":
        freq = int(value)
        if freq > 3:
            return f"The patient filed {freq} claims in the last 30 days, which is unusually frequent and can raise caution."
        return f"The patient filed {freq} claims in the last 30 days, which looks normal."

    if feature_name == "distance_to_provider":
        distance = value
        if distance > 50:
            return f"The patient traveled {distance:.1f} miles to this provider, which is farther than usual and may deserve verification."
        return f"The patient traveled {distance:.1f} miles to this provider, which looks normal."

    return f"The feature {feature_name} had a measurable effect on the decision, with a value of {value:.4f}."


def _get_drug_disease_match_score(db: Session, data: dict, claim_id: str) -> float:
    try:
        from .knowledge_service import get_claim_knowledge
        provider = data.get("provider_name")
        company = data.get("insurance_company")
        diagnosis = data.get("diagnosis_code")

        if provider:
            result = get_claim_knowledge("provider", provider, "drug_disease_match_percent")
            if result and result.get("value") is not None:
                return float(result.get("value") or 0.7)

        if company:
            result = get_claim_knowledge("insurance_company", company, "drug_disease_match_percent")
            if result and result.get("value") is not None:
                return float(result.get("value") or 0.7)

        if diagnosis:
            result = get_claim_knowledge("diagnosis", diagnosis, "drug_disease_match_percent")
            if result and result.get("value") is not None:
                return float(result.get("value") or 0.7)

        return 0.7
    except Exception:
        return 0.7


def _get_provider_fraud_history(db: Session, provider_name: str) -> dict:
    try:
        from ..core.database_user import DB_PATH
        engine = create_engine(f"sqlite:///{DB_PATH}")
        SessionLocal = sessionmaker(bind=engine)
        session = SessionLocal()
        result = session.execute(text("""
            SELECT 
                COUNT(*) as total_claims,
                AVG(CASE WHEN is_fraud = 1 THEN 1 ELSE 0 END) as fraud_rate
            FROM claims
            WHERE hospital_name = :provider OR provider_name = :provider
        """), {"provider": provider_name})
        row = result.fetchone()
        session.close()
        return {
            "total_claims": row[0] or 0 if row else 0,
            "fraud_rate": row[1] or 0.0 if row else 0.0
        }
    except Exception:
        return {"total_claims": 0, "fraud_rate": 0.0}


def _check_diagnosis_procedure_match(db: Session, diagnosis_code: str, procedure_code: str) -> bool:
    try:
        from .knowledge_service import vector_search_sqlite
        query = f"{diagnosis_code} procedure {procedure_code}"
        results = vector_search_sqlite(db, query, diagnosis_code=diagnosis_code, top_k=1, threshold=0.75)
        return len(results) > 0 and results[0].get("similarity", 0) > 0.8
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
            "explanation": "Pre-authorization was confirmed before treatment, which supports the claim."
        })
    else:
        factors.append({
            "feature": "pre_auth",
            "value": 0,
            "contribution": 0.30,
            "direction": "increases_risk",
            "explanation": "No pre-authorization was found before treatment, which makes the claim less certain."
        })

    factors.append({
        "feature": "claim_amount",
        "value": amount,
        "contribution": 0.28 if amount > 5000 else 0.08,
        "direction": "increases_risk" if amount > 5000 else "decreases_risk",
        "explanation": f"Claim amount of ${amount:,.2f} is {'high' if amount > 5000 else 'moderate'} for this case."
    })

    if ratio > 8:
        factors.append({
            "feature": "amount_income_ratio",
            "value": round(ratio, 2),
            "contribution": 0.20,
            "direction": "increases_risk",
            "explanation": "The claim amount is unusually large compared with the patient’s income."
        })
    else:
        factors.append({
            "feature": "amount_income_ratio",
            "value": round(ratio, 2),
            "contribution": -0.10,
            "direction": "decreases_risk",
            "explanation": "The claim amount looks proportionate to the patient’s income."
        })

    if age >= 65:
        factors.append({
            "feature": "age",
            "value": age,
            "contribution": -0.15,
            "direction": "decreases_risk",
            "explanation": f"The patient is {age}, which often follows a more stable claim pattern."
        })

    return sorted(factors, key=lambda x: abs(x["contribution"]), reverse=True)