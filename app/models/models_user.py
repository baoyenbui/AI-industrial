from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean, Index, JSON
from sqlalchemy.sql import func
from ..core.database_user import Base

class Claim(Base):
    __tablename__ = "claims"

    id = Column(Integer, primary_key=True, index=True)
    claim_id = Column(String(36), unique=True, index=True, nullable=False)

    patient_age = Column(Integer)
    patient_gender = Column(String(20))
    patient_income = Column(Float)
    patient_employment_status = Column(String(100))
    patient_marital_status = Column(String(50))

    policy_number = Column(String(100))
    provider_name = Column(String(200), index=True)
    provider_type = Column(String(50))
    insurance_company = Column(String(200), index=True)
    hospital_name = Column(String(200), nullable=True)

    date_of_service = Column(String(50))
    pre_authorization_status = Column(String(10))
    claim_type = Column(String(50))
    diagnosis_code = Column(String(50), index=True)
    procedure_code = Column(String(50))
    medicine_name = Column(String(200))

    claim_amount = Column(Float, nullable=False)
    claim_status = Column(String(20), default="Pending")
    reimbursement_amount = Column(Float, default=0.0)
    decision = Column(String(20))
    confidence = Column(Float, default=0.75)
    rule_applied = Column(String(50), default="default")

    medicine_ok = Column(Boolean)
    fraud_score = Column(Float, default=0.0)
    fraud_label = Column(String(20))
    fraud_flags = Column(JSON)

    coverage_percent = Column(Float)
    baseline_amount = Column(Float)
    knowledge_source = Column(String(50), default="initial")
    knowledge_confidence = Column(Float, default=0.5)
    knowledge_count = Column(Integer, default=1)

    uploaded_files = Column(JSON)
    raw_ocr_text = Column(Text, nullable=True)
    user_verified = Column(Boolean, default=False)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    __table_args__ = (
        Index("ix_claim_provider", "provider_name"),
        Index("ix_claim_company", "insurance_company"),
        Index("ix_claim_decision", "decision"),
        Index("ix_claim_knowledge", "knowledge_source", "knowledge_confidence"),
    )

class ClaimAuditLog(Base):
    __tablename__ = "claim_audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    claim_id = Column(String(36), index=True, nullable=False)
    action = Column(String(50), nullable=False)
    user_id = Column(String(50), nullable=True)
    ip_address = Column(String(50), nullable=True)
    user_agent = Column(String(200), nullable=True)
    details = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())