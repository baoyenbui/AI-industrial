# app/services/shap_service.py
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
        except:
            return None
    return None


def explain_decision(data: dict, fraud_result: dict = None) -> List[Dict]:
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
                "direction": "increases_risk" if shap_val > 0 else "decreases_risk"
            })

        contributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)
        return contributions[:8]   
    except:
        return _fallback_explanation(data)


def _get_explainer(model):
    global _explainer
    if _explainer is None:
        import shap
        _explainer = shap.TreeExplainer(model)
    return _explainer


def _fallback_explanation(data: dict) -> List[Dict]:
    """Giải thích rule-based khi không có model"""
    amount = float(data.get("claim_amount") or 0)
    pre_auth = str(data.get("pre_auth", "")).lower() == "yes"
    ratio = amount / float(data.get("patient_income") or 1)

    return [
        {"feature": "claim_amount", "value": amount, "contribution": 0.32 if amount > 20000 else 0.12, "direction": "increases_risk"},
        {"feature": "pre_auth", "value": pre_auth, "contribution": -0.25 if pre_auth else 0.28, "direction": "decreases_risk" if pre_auth else "increases_risk"},
        {"feature": "amount_income_ratio", "value": round(ratio, 2), "contribution": 0.22 if ratio > 8 else -0.08, "direction": "increases_risk" if ratio > 8 else "decreases_risk"},
    ]


def format_shap_for_prompt(shap_list: List[Dict]) -> str:
    if not shap_list:
        return ""
    lines = ["DECISION EXPLANATION (Feature Contributions):"]
    for item in shap_list[:6]:
        sign = "↑ Risk" if item["direction"] == "increases_risk" else "↓ Risk"
        lines.append(f"- {item['feature']}: {item['value']} → {sign} ({item['contribution']:+.3f})")
    return "\n".join(lines)