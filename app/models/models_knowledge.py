from sqlalchemy import Column, Integer, String, Float, Text, DateTime, JSON, Index
from sqlalchemy.sql import func
import uuid
from ..core.database_knowledge import KnowledgeBase


class MedicalKnowledge(KnowledgeBase):
    __tablename__ = "medical_knowledge"

    id   = Column(Integer, primary_key=True, autoincrement=True)
    uuid = Column(String(36), unique=True, default=lambda: str(uuid.uuid4()), index=True)

    diagnosis_code = Column(String(50), index=True)
    description    = Column(Text, nullable=False)
    category       = Column(String(50), index=True)

    quantity           = Column(Float)
    unit_price_low     = Column(Float)
    unit_price_high    = Column(Float)
    typical_total_low  = Column(Float)
    typical_total_high = Column(Float)

    medical_code      = Column(String(50))
    drug_generic_name = Column(String(200))
    manufacturer      = Column(String(150))

    # is_appropriate: "yes" | "no" | "conditional" | "unknown"
    # Không có review_status column — is_appropriate đóng vai trò đó.
    # vector_search filter theo is_appropriate != "no", không dùng review_status.
    is_appropriate       = Column(String(20), default="unknown", index=True)
    appropriateness_note = Column(Text)
    confidence           = Column(Float, default=0.65)

    source           = Column(String(100), default="ocr_extracted")
    source_reference = Column(Text)
    hospital_code    = Column(String(50), index=True)
    region           = Column(String(50))

    created_at  = Column(DateTime(timezone=True), server_default=func.now())
    updated_at  = Column(DateTime(timezone=True), onupdate=func.now())
    reviewed_by = Column(String(100))
    reviewed_at = Column(DateTime(timezone=True))

    chunk_hash      = Column(String(64), unique=True, index=True)
    embedding_model = Column(String(80), default="all-MiniLM-L6-v2")

    # sample_count KHÔNG có column riêng — lưu trong metadata_json["sample_count"].
    # embedding   KHÔNG có column riêng — lưu trong metadata_json["embedding"].
    # Lý do: SQLite không có pgvector, tránh migration phức tạp.
    metadata_json = Column(JSON)

    __table_args__ = (
        Index("ix_diag_category", "diagnosis_code", "category"),
        # Bỏ confidence.desc() — functional index không hợp lệ trên SQLite,
        # gây crash khi create_all(). Dùng plain index thay thế.
        Index("ix_confidence",    "confidence"),
        Index("ix_source",        "source"),
    )


class KnowledgeHistory(KnowledgeBase):
    __tablename__ = "medical_knowledge_history"

    id             = Column(Integer, primary_key=True, autoincrement=True)
    knowledge_uuid = Column(String(36), index=True)
    action         = Column(String(30))
    changed_by     = Column(String(100))
    changed_at     = Column(DateTime(timezone=True), server_default=func.now())
    old_values     = Column(JSON)
    new_values     = Column(JSON)
    reason         = Column(Text)


class InvoiceExtractionLog(KnowledgeBase):
    __tablename__ = "invoice_extraction_logs"

    id                    = Column(Integer, primary_key=True, autoincrement=True)
    ocr_text_hash         = Column(String(64), index=True)
    raw_ocr_text          = Column(Text)
    extracted_json        = Column(JSON)
    diagnosis_code        = Column(String(50))
    total_billed          = Column(Float)
    extraction_confidence = Column(Float)
    processed_by_model    = Column(String(100))
    created_at            = Column(DateTime(timezone=True), server_default=func.now())