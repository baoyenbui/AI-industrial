import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sqlite3
from sklearn.model_selection import train_test_split
import lightgbm as lgb

from app.services.fraud_service import build_features

conn = sqlite3.connect("health_claims.db")

df = pd.read_sql("""
    SELECT 
        claim_amount,
        patient_age,
        patient_income,
        pre_authorization_status,
        claim_type,
        diagnosis_code,
        decision
    FROM claims 
    WHERE claim_amount IS NOT NULL AND claim_amount > 0
""", conn)

if len(df) < 3:
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    joblib.dump(None, model_dir / "fraud_model_lgb.pkl")
    exit()

df["label"] = (df["decision"] == "Denied").astype(int)

def prepare_features(row):
    data = {
        "claim_amount": row["claim_amount"],
        "patient_age": row.get("patient_age"),
        "patient_income": row.get("patient_income"),
        "pre_auth": row.get("pre_authorization_status"),
        "claim_type": row.get("claim_type"),
        "provider_specialty": "",
        "diagnosis_code": row.get("diagnosis_code"),
    }
    return build_features(data)[0]

X = np.array([prepare_features(row) for _, row in df.iterrows()])
y = df["label"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y if len(set(y)) > 1 else None
)

model = lgb.LGBMClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=5,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    class_weight='balanced',
    verbose=-1
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

model_dir = Path("models")
model_dir.mkdir(exist_ok=True)
joblib.dump(model, model_dir / "fraud_model_lgb.pkl")