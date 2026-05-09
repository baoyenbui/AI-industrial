import numpy as np
import joblib
from pathlib import Path
from typing import Dict, List

MODEL_PATH = Path(__file__).parent.parent / "models" / "fraud_model_lgb.pkl"

HIGH_RISK_DIAGNOSIS = {"C50", "C61", "C34", "E11", "I21", "I25", "C18", "C19"}

CLAIM_TYPE_MAP = {
    "medical": 0, "dental": 1, "vision": 2, "pharmacy": 3,
    "mental health": 4, "maternity": 5, "surgery": 6
}

SPECIALTY_MAP = {
    "cardiology": 0, "oncology": 1, "orthopedics": 2, "neurology": 3,
    "radiology": 4, "surgery": 5, "emergency": 6, "internal medicine": 7,
    "general": 8
}

SUBMISSION_MAP = {
    "online": 0, "electronic": 0, "paper": 1, "hospital": 2,
    "fax": 3, "mail": 4, "portal": 0
}

FEATURE_NAMES = [
    "claim_amount", "patient_age", "patient_income", "pre_auth",
    "amount_income_ratio", "is_high_risk_diagnosis", "diagnosis_prefix",
    "claim_type", "provider_specialty", "submission_method"
]

_model = None


def _load_model():
    global _model
    if _model is None and MODEL_PATH.exists():
        try:
            _model = joblib.load(MODEL_PATH)
        except:
            _model = None
    return _model


def build_features(data: dict) -> np.ndarray:
    amount = float(data.get("claim_amount") or 0)
    age = float(data.get("age") or data.get("patient_age") or 0)
    income = float(data.get("patient_income") or 1.0)
    pre_auth = 1.0 if str(data.get("pre_auth", "")).strip().lower() == "yes" else 0.0

    ratio = amount / income if income > 0 else 999.0
    diag = str(data.get("diagnosis_code", "")).strip().upper()
    is_high_risk = 1.0 if any(code in diag for code in HIGH_RISK_DIAGNOSIS) else 0.0
    diag_prefix = ord(diag[0]) - 65 if diag and diag[0].isalpha() else 0

    claim_type = CLAIM_TYPE_MAP.get(str(data.get("claim_type", "")).lower(), 0)
    specialty = SPECIALTY_MAP.get(str(data.get("provider_specialty", "")).lower(), 8)
    submission = SUBMISSION_MAP.get(str(data.get("claim_submission_method", "")).lower(), 1)

    return np.array([[
        amount, age, income, pre_auth, ratio, is_high_risk,
        diag_prefix, claim_type, specialty, submission
    ]], dtype=np.float32)


def predict_fraud(data: dict) -> Dict:
    features = build_features(data)
    model = _load_model()

    if model is not None:
        try:
            prob = model.predict_proba(features)[0][1]
            score = float(prob)
        except:
            score = _rule_based_score(data)
    else:
        score = _rule_based_score(data)

    flags = _get_fraud_flags(data, score)
    score = min(1.0, score + 0.08 * len(flags))

    label = "HIGH_RISK" if score >= 0.65 else "MEDIUM_RISK" if score >= 0.40 else "LOW_RISK"

    return {
        "fraud_score": round(score, 4),
        "label": label,
        "flags": flags,
        "risk_level": label
    }


def _rule_based_score(data: dict) -> float:
    score = 0.0
    amount = float(data.get("claim_amount") or 0)
    income = float(data.get("patient_income") or 1)
    pre_auth = str(data.get("pre_auth", "")).lower()

    if amount > 50000:
        score += 0.35
    elif amount > 20000:
        score += 0.18
    if pre_auth != "yes" and amount > 8000:
        score += 0.28
    if income > 0 and amount / income > 8:
        score += 0.22

    return min(score, 0.92)


def _get_fraud_flags(data: dict, score: float) -> List[str]:
    flags = []
    amount = float(data.get("claim_amount") or 0)
    income = float(data.get("patient_income") or 1)
    pre_auth = str(data.get("pre_auth", "")).lower()

    if amount > 45000:
        flags.append(f"Extremely high claim amount (${amount:,.0f})")
    if pre_auth != "yes" and amount > 7000:
        flags.append("High value claim without pre-authorization")
    if income > 0 and amount / income > 10:
        flags.append(f"Claim is {amount/income:.1f}x monthly income")
    if not data.get("diagnosis_code"):
        flags.append("Missing diagnosis code")

    return flags