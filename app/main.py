from fastapi import FastAPI, UploadFile, File
from app.claim_model import get_claim_approval, ocr_pipeline

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Health Claim API running"}

@app.get("/predict")
def predict(query: str):
    return get_claim_approval(query)

@app.post("/ocr-claim")
def ocr_claim(file: UploadFile = File(...)):
    return ocr_pipeline(file)