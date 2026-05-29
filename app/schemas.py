from pydantic import BaseModel, Field, field_validator
from typing import Optional
import re


class ClaimInput(BaseModel):
    PatientAge:    Optional[int]   = Field(default=0, ge=0, le=120)
    PatientGender: Optional[str]   = "Other"
    PatientIncome: Optional[float] = Field(default=0.0, ge=0)

    PatientEmploymentStatus: Optional[str] = "unknown"
    PatientMaritalStatus:    Optional[str] = "unknown"

    ProviderSpecialty: Optional[str] = "unknown"
    ClaimType:         Optional[str] = "unknown"

    DiagnosisCode: Optional[str]   = "UNKNOWN"
    ProcedureCode: Optional[str]   = "UNKNOWN"
    ClaimAmount:   Optional[float] = Field(default=0.0, ge=0)

    PolicyNumber:  Optional[str] = ""
    DateOfService: Optional[str] = ""
    HospitalName:  Optional[str] = ""

    # KHÔNG dùng Literal — frontend có thể gửi "Choose an option", None, ""
    # claim_service sẽ normalize qua _resolve_pre_auth
    PreAuthorizationStatus: Optional[str] = "No"

    ClaimSubmissionMethod: Optional[str] = "unknown"
    ClaimStatus:           Optional[str] = "pending"

    # ------------------------------------------------------------------ #
    # Validators
    # ------------------------------------------------------------------ #

    @field_validator("PatientGender", mode="before")
    @classmethod
    def _gender(cls, v):
        if not v: return "Other"
        vl = str(v).lower()
        if "female" in vl:               return "Female"
        if "male" in vl:                 return "Male"
        if vl in ("other", "choose the gender", ""): return "Other"
        return "Other"

    @field_validator("PatientEmploymentStatus", mode="before")
    @classmethod
    def _emp(cls, v):
        if not v: return "unknown"
        vl = str(v).lower()
        if any(x in vl for x in ["self", "freelance", "contract"]): return "self-employed"
        if any(x in vl for x in ["employed", "working"]):           return "employed"
        if any(x in vl for x in ["unemployed", "jobless"]):         return "unemployed"
        if "student"  in vl: return "student"
        if "retired"  in vl: return "retired"
        return str(v).strip() or "unknown"

    @field_validator("PatientMaritalStatus", mode="before")
    @classmethod
    def _marital(cls, v):
        if not v: return "unknown"
        vl = str(v).lower()
        if any(x in vl for x in ["single", "unmarried"]): return "single"
        if "married"  in vl: return "married"
        if "divorced" in vl: return "divorced"
        if any(x in vl for x in ["widow", "widower"]):    return "widowed"
        return str(v).strip() or "unknown"

    @field_validator("PreAuthorizationStatus", mode="before")
    @classmethod
    def _pre_auth(cls, v):
        if not v: return "No"
        vs = str(v).strip().lower()
        if vs in ("yes", "true", "1", "approved", "granted"): return "Yes"
        if vs in ("no",  "false","0", "denied",   "not required"): return "No"
        # "choose an option", empty, unknown → None signal to claim_service
        return "No"

    @field_validator("PolicyNumber", mode="before")
    @classmethod
    def _policy(cls, v):
        if not v: return ""
        return str(v).strip().upper()

    @field_validator("DateOfService", mode="before")
    @classmethod
    def _date(cls, v):
        if not v: return ""
        return str(v).strip()

    @field_validator("HospitalName", mode="before")
    @classmethod
    def _hospital(cls, v):
        if not v: return ""
        return str(v).strip()

    @field_validator("DiagnosisCode", "ProcedureCode", "ProviderSpecialty", "ClaimType", mode="before")
    @classmethod
    def _clean(cls, v):
        if not v: return "UNKNOWN"
        v = str(v).strip()
        return v if len(v) >= 2 else "UNKNOWN"

    @field_validator("ClaimSubmissionMethod", "ClaimStatus", mode="before")
    @classmethod
    def _lower_str(cls, v):
        if not v: return "unknown"
        return str(v).strip().lower()