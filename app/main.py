from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from app.claim_model import get_claim_approval, ocr_pipeline
from app.schemas import ClaimInput

app = FastAPI()


@app.get("/")
def home():
    return {"status": "ok"}


def build_query(data: dict):
    def norm(v):
        if v is None or v == "":
            return "unknown"
        return str(v)

    return f"""
Age:{norm(data.get('PatientAge'))}
Gender:{norm(data.get('PatientGender'))}
Income:{norm(data.get('PatientIncome'))}
Employment:{norm(data.get('PatientEmploymentStatus'))}
Specialty:{norm(data.get('ProviderSpecialty'))}
Type:{norm(data.get('ClaimType'))}
Diagnosis:{norm(data.get('DiagnosisCode'))}
Procedure:{norm(data.get('ProcedureCode'))}
Amount:{norm(data.get('ClaimAmount'))}
""".strip()


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = []

    for e in exc.errors():
        field = ".".join(str(x) for x in e["loc"] if x != "body")
        errors.append({
            "field": field,
            "message": e["msg"]
        })

    return JSONResponse(
        status_code=200,
        content={
            "status": "invalid_input",
            "decision": "Pending",
            "confidence": 0.0,
            "reason": "validation_failed",
            "errors": errors
        }
    )

@app.post("/predict")
def predict(data: ClaimInput):
    try:
        clean_data = data.model_dump()
        query = build_query(clean_data)
        result = get_claim_approval(query)

        return {
            "status": "ok",
            **result
        }

    except Exception as e:
        return {
            "status": "system_error",
            "decision": "Pending",
            "confidence": 0.0,
            "reason": "backend_exception"
        }

@app.post("/ocr-claim")
def ocr_claim(file: UploadFile = File(...)):
    try:
        result = ocr_pipeline(file)

        if result.get("status") != "ok":
            return {
                "status": "ocr_failed",
                "extracted_data": {},
                "raw_text": ""
            }

        return result

    except Exception:
        return {
            "status": "ocr_failed",
            "extracted_data": {},
            "raw_text": ""
        }