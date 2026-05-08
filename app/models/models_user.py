from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean
from sqlalchemy.sql import func
from ..core.database_user import Base

class Claim(Base):
    __tablename__ = "claims"
    
    id = Column(Integer, primary_key=True, index=True)
    claim_id = Column(String, unique=True, index=True, nullable=False)
    patient_age = Column(Integer)
    patient_gender = Column(String(20))
    patient_income = Column(Float)
    patient_employment_status = Column(String(50))
    patient_marital_status = Column(String(30))
    policy_number = Column(String(50), index=True)
    date_of_service = Column(String(20))
    hospital_name = Column(String(100))
    pre_authorization_status = Column(String(10))
    claim_type = Column(String(50))
    diagnosis_code = Column(String(20))
    procedure_code = Column(String(20))
    claim_amount = Column(Float)
    claim_status = Column(String(20), default="Pending")
    reimbursement_amount = Column(Float, default=0.0)
    decision = Column(String(20))
    confidence = Column(Float)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    raw_ocr_text = Column(Text, nullable=True)

class ClaimAuditLog(Base):
    __tablename__ = "claim_audit_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    claim_id = Column(String, index=True, nullable=False)
    action = Column(String(50), nullable=False)
    user_id = Column(String(50), nullable=True)
    ip_address = Column(String(50), nullable=True)
    user_agent = Column(String(200), nullable=True)
    details = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class Hospital(Base):
    __tablename__ = "hospitals"
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), unique=True, index=True)
    is_trusted = Column(Boolean, default=True)
    created_at = Column(DateTime, server_default=func.now())