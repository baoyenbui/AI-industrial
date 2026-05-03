from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator, ValidationError
from typing import Literal, Optional
import re

app = FastAPI()


class ClaimInput(BaseModel):
    PatientAge: Optional[int] = Field(default=0, ge=0, le=120)
    PatientGender: Optional[str] = "Other"
    PatientIncome: Optional[float] = Field(default=0.0, ge=0)

    PatientEmploymentStatus: Optional[str] = "unknown"

    ProviderSpecialty: Optional[str] = "unknown"
    ClaimType: Optional[str] = "unknown"

    DiagnosisCode: Optional[str] = "UNKNOWN"
    ProcedureCode: Optional[str] = "UNKNOWN"

    ClaimAmount: Optional[float] = Field(default=0.0, ge=0)

    @field_validator("PatientGender", mode="before")
    @classmethod
    def gender(cls, v):
        if not v:
            return "Other"
        v = str(v).lower()

        if "male" in v and "female" not in v:
            return "Male"
        if "female" in v:
            return "Female"
        return "Other"

    @field_validator("PatientEmploymentStatus", mode="before")
    @classmethod
    def emp(cls, v):
        if not v:
            return "unknown"

        v = str(v).lower()

        if any(x in v for x in ["employed", "working"]):
            return "employed"
        if any(x in v for x in ["unemployed", "jobless"]):
            return "unemployed"
        if any(x in v for x in ["self", "freelance", "contract"]):
            return "self-employed"
        if "student" in v:
            return "student"
        if "retired" in v:
            return "retired"

        return "unknown"

    @field_validator("DiagnosisCode", "ProcedureCode", "ProviderSpecialty", "ClaimType", mode="before")
    @classmethod
    def clean_text(cls, v):
        if not v:
            return "UNKNOWN"

        v = str(v).strip()

        if len(v) < 2:
            return "UNKNOWN"

        return v


@app.exception_handler(ValidationError)
async def validation_exception_handler(request, exc):
    errors = []
    for e in exc.errors():
        errors.append({
            "field": e["loc"][-1],
            "message": e["msg"]
        })

    return {
        "status": "failed",
        "error_type": "validation_error",
        "errors": errors
    }


@app.post("/predict")
def predict(data: ClaimInput):
    try:
        if data.PatientIncome == 0:
            return {
                "decision": "Denied",
                "reason": "Income is 0",
                "confidence": 0.99
            }

        if not data.DiagnosisCode or data.DiagnosisCode == "UNKNOWN":
            return {
                "decision": "Pending",
                "reason": "Missing diagnosis",
                "confidence": 0.6
            }

        return {
            "decision": "Approved",
            "reason": "All conditions satisfied",
            "confidence": 0.95
        }

    except Exception as e:
        return {
            "status": "failed",
            "error_type": "internal_error",
            "message": str(e)
        }