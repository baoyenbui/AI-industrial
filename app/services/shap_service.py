import numpy as np
import joblib
from pathlib import Path
from typing import List, Dict

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


def explain_decision(data: dict, fraud_result: dict = None) -> List[Dict]:
    model = _load_model()
    if model is None:
        return _fallback_explanation(data)

    try:
        features  = build_features(data)[0]
        explainer = _get_explainer(model)
        shap_values = explainer.shap_values(features.reshape(1, -1))

        if isinstance(shap_values, list):
            values = shap_values[1][0]
        else:
            values = shap_values[0]

        contributions = []
        for name, value, shap_val in zip(FEATURE_NAMES, features, values):
            contributions.append({
                "feature":      name,
                "value":        round(float(value), 4),
                "contribution": round(float(shap_val), 4),
                "direction":    "increases_risk" if shap_val > 0 else "decreases_risk",
            })

        contributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)
        return contributions[:8]

    except Exception:
        return _fallback_explanation(data)


def _get_explainer(model):
    global _explainer
    if _explainer is None:
        import shap
        _explainer = shap.TreeExplainer(model)
    return _explainer


def _fallback_explanation(data: dict) -> List[Dict]:
    amount   = float(data.get("claim_amount") or 0)
    pre_auth = str(data.get("pre_auth", "")).strip().lower() == "yes"
    income   = float(data.get("patient_income") or 1)
    age      = int(data.get("age") or 0)
    ratio    = amount / income if income > 0 else 0

    factors = []

    if pre_auth:
        factors.append({
            "feature": "pre_auth", "value": 1,
            "contribution": -0.30, "direction": "decreases_risk",
        })
    else:
        factors.append({
            "feature": "pre_auth", "value": 0,
            "contribution": 0.30, "direction": "increases_risk",
        })

    factors.append({
        "feature": "claim_amount", "value": amount,
        "contribution": 0.28 if amount > 5000 else 0.08,
        "direction": "increases_risk" if amount > 5000 else "decreases_risk",
    })

    if ratio > 8:
        factors.append({
            "feature": "amount_income_ratio", "value": round(ratio, 2),
            "contribution": 0.20, "direction": "increases_risk",
        })
    else:
        factors.append({
            "feature": "amount_income_ratio", "value": round(ratio, 2),
            "contribution": -0.10, "direction": "decreases_risk",
        })

    if age >= 65:
        factors.append({
            "feature": "age", "value": age,
            "contribution": -0.15, "direction": "decreases_risk",
        })

    return sorted(factors, key=lambda x: abs(x["contribution"]), reverse=True)