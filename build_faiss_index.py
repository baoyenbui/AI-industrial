import os
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "health_insurance_claims.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
INDEX_PATH = os.path.join(MODEL_DIR, "claims_faiss.index")

os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

columns_needed = [
    "PatientAge",
    "PatientGender",
    "PatientIncome",
    "PatientMaritalStatus",
    "PatientEmploymentStatus",
    "ProviderSpecialty",
    "ClaimAmount",
    "ClaimType",
    "ClaimSubmissionMethod",
    "DiagnosisCode",
    "ProcedureCode",
    "ClaimStatus"
]

df = df[columns_needed].dropna()

def format_row(row):
    return (
        f"Age: {row['PatientAge']}, Gender: {row['PatientGender']}, "
        f"Income: {row['PatientIncome']}, Marital: {row['PatientMaritalStatus']}, "
        f"Employment: {row['PatientEmploymentStatus']}, "
        f"Specialty: {row['ProviderSpecialty']}, "
        f"ClaimType: {row['ClaimType']}, Submission: {row['ClaimSubmissionMethod']}, "
        f"Diagnosis: {row['DiagnosisCode']}, Procedure: {row['ProcedureCode']}, "
        f"Amount: {row['ClaimAmount']}"
    )

df["ClaimText"] = df.apply(format_row, axis=1)

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(df["ClaimText"].tolist(), show_progress_bar=True)
embeddings = np.array(embeddings, dtype="float32")

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

faiss.write_index(index, INDEX_PATH)

print("DONE. Saved index at:", INDEX_PATH)