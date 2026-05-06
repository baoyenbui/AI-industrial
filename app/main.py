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
            return ""
        return str(v).strip()

    return f"""
PatientAge: {norm(data.get('PatientAge'))}
PatientGender: {norm(data.get('PatientGender'))}
PatientIncome: {norm(data.get('PatientIncome'))}
PatientEmploymentStatus: {norm(data.get('PatientEmploymentStatus'))}
ProviderSpecialty: {norm(data.get('ProviderSpecialty'))}
ClaimType: {norm(data.get('ClaimType'))}
DiagnosisCode: {norm(data.get('DiagnosisCode'))}
ProcedureCode: {norm(data.get('ProcedureCode'))}
ClaimAmount: {norm(data.get('ClaimAmount'))}
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