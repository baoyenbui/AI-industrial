from fastapi import FastAPI, HTTPException
from jsonschema import ValidationError
from pydantic import BaseModel, Field, field_validator
from typing import Literal, Optional
import re

app = FastAPI()


class ClaimInput(BaseModel):
    PatientAge: Optional[int] = Field(default=0, ge=0, le=120)
    PatientGender: Optional[str] = "Other"
    PatientIncome: Optional[float] = Field(default=0.0, ge=0)

    PatientEmploymentStatus: Optional[str] = "unknown"
    PatientMaritalStatus: Optional[str] = "unknown"

    ProviderSpecialty: Optional[str] = "unknown"
    ClaimType: Optional[str] = "unknown"

    DiagnosisCode: Optional[str] = "UNKNOWN"
    ProcedureCode: Optional[str] = "UNKNOWN"

    ClaimAmount: Optional[float] = Field(default=0.0, ge=0)

    PolicyNumber: Optional[str] = Field(default="", alias="PolicyNumber")
    DateOfService: Optional[str] = Field(default="", alias="DateOfService")
    HospitalName: Optional[str] = Field(default="", alias="HospitalName")
    PreAuthorizationStatus: Optional[Literal["Yes", "No"]] = "No"

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

    @field_validator("PatientMaritalStatus", mode="before")
    @classmethod
    def marital(cls, v):
        if not v:
            return "unknown"
        v = str(v).lower()
        if any(x in v for x in ["single", "unmarried"]):
            return "single"
        if "married" in v:
            return "married"
        if "divorced" in v:
            return "divorced"
        if any(x in v for x in ["widow", "widower"]):
            return "widowed"
        return "unknown"

    @field_validator("PolicyNumber", mode="before")
    @classmethod
    def policy_number(cls, v):
        if not v:
            return ""
        return str(v).strip().upper()

    @field_validator("DateOfService", mode="before")
    @classmethod
    def date_of_service(cls, v):
        if not v:
            return ""
        return str(v).strip()

    @field_validator("HospitalName", mode="before")
    @classmethod
    def hospital_name(cls, v):
        if not v:
            return ""
        return str(v).strip()

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
        if data.PatientAge is None or data.PatientAge == 0:
            return {"decision": "Pending", "reason": "Missing patient age", "confidence": 0.6}

        if data.ClaimAmount is None or data.ClaimAmount <= 0:
            return {"decision": "Denied", "reason": "Invalid claim amount", "confidence": 0.9}

        if not data.PolicyNumber:
            return {
                "decision": "Pending",
                "reason": "Policy number is missing (will be reviewed manually)",
                "confidence": 0.65,
                "needs_manual_review": True
            }

        if data.ClaimAmount > 50000000 and not data.PreAuthorizationStatus == "Yes":
            return {"decision": "Pending", "reason": "High amount without pre-authorization", "confidence": 0.7}

        return {
            "decision": "Approved",
            "reason": "All conditions satisfied",
            "confidence": 0.92,
            "policy_number": data.PolicyNumber,
            "hospital_name": data.HospitalName
        }

    except Exception as e:
        return {
            "status": "failed",
            "error_type": "internal_error",
            "message": str(e)
        }