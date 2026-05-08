import hashlib
from datetime import datetime
from typing import Optional
from sqlalchemy.orm import Session

from ..models.models_knowledge import MedicalKnowledge, KnowledgeHistory, InvoiceExtractionLog


PRICE_TOLERANCE        = 0.25
MIN_SAMPLES_AUTO       = 3
AUTO_APPROVE_CONFIDENCE = 0.88


def _hash(text: str) -> str:
    return hashlib.sha256(text.strip().lower().encode("utf-8")).hexdigest()[:64]


def _log_history(
    db: Session,
    knowledge_uuid: str,
    action: str,
    changed_by: str = "system",
    old_values: dict = None,
    new_values: dict = None,
    reason: str = None,
):
    db.add(KnowledgeHistory(
        knowledge_uuid=knowledge_uuid,
        action=action,
        changed_by=changed_by,
        old_values=old_values or {},
        new_values=new_values or {},
        reason=reason,
    ))


def _price_in_range(new_price: float, record: MedicalKnowledge) -> bool:
    # Dùng midpoint(low, high) làm tham chiếu — nhất quán với cách vector_search trả về unit_price_avg
    low  = record.unit_price_low
    high = record.unit_price_high
    if not new_price:
        return True
    if not low and not high:
        return True
    avg = ((low or 0) + (high or low or 0)) / 2
    if avg == 0:
        return True
    return abs(new_price - avg) / avg <= PRICE_TOLERANCE


def upsert_knowledge_item(
    db: Session,
    item: dict,
    diagnosis_code: str = None,
    source: str = "ocr_extracted",
) -> tuple[Optional[MedicalKnowledge], str]:
    description = str(item.get("description") or "").strip()
    if not description:
        return None, "skipped"

    chunk_hash = _hash(description)
    existing   = db.query(MedicalKnowledge).filter_by(chunk_hash=chunk_hash).first()

    new_price = item.get("unit_price") or item.get("total")
    new_conf  = float(item.get("confidence") or 0.65)

    if existing:
        old_vals = {
            "unit_price_low":  existing.unit_price_low,
            "unit_price_high": existing.unit_price_high,
            "confidence":      existing.confidence,
            "is_appropriate":  existing.is_appropriate,
        }

        if new_price:
            existing.unit_price_low  = min(existing.unit_price_low,  new_price) if existing.unit_price_low  else new_price
            existing.unit_price_high = max(existing.unit_price_high, new_price) if existing.unit_price_high else new_price

        meta = existing.metadata_json or {}
        count = meta.get("sample_count", 1) + 1
        meta["sample_count"] = count
        existing.metadata_json = meta

        if (
            existing.is_appropriate == "unknown"
            and count >= MIN_SAMPLES_AUTO
            and new_conf >= AUTO_APPROVE_CONFIDENCE
            and _price_in_range(new_price, existing)
        ):
            existing.is_appropriate = "yes"
            existing.confidence     = min(0.95, new_conf + 0.05)
            _log_history(
                db, existing.uuid, "auto_approved",
                old_values=old_vals,
                new_values={"is_appropriate": "yes", "confidence": existing.confidence},
                reason=f"Seen {count}x with consistent pricing",
            )
        else:
            _log_history(
                db, existing.uuid, "updated",
                old_values=old_vals,
                new_values={
                    "unit_price_low":  existing.unit_price_low,
                    "unit_price_high": existing.unit_price_high,
                },
            )

        db.commit()
        return existing, "updated"

    embedding = _encode(description)

    record = MedicalKnowledge(
        chunk_hash         = chunk_hash,
        diagnosis_code     = diagnosis_code,
        description        = description,
        category           = item.get("category"),
        medical_code       = item.get("code"),
        unit_price_low     = new_price,
        unit_price_high    = new_price,
        typical_total_low  = item.get("total"),
        typical_total_high = item.get("total"),
        confidence         = new_conf,
        source             = source,
        is_appropriate     = "unknown",
        embedding_model    = "all-MiniLM-L6-v2",
        metadata_json      = {
            "sample_count":  1,
            "embedding":     embedding,
            "original_item": item,
        },
    )

    db.add(record)
    db.flush()

    _log_history(
        db, record.uuid, "created",
        changed_by=source,
        new_values={"description": description, "confidence": new_conf},
    )

    db.commit()
    return record, "created"


def expert_review(
    db: Session,
    knowledge_id: int,
    decision: str,
    note: str = None,
    confidence: float = 0.95,
    reviewed_by: str = "expert",
) -> bool:
    record = db.query(MedicalKnowledge).filter_by(id=knowledge_id).first()
    if not record:
        return False

    if decision not in ("yes", "no", "conditional"):
        return False

    old_vals = {
        "is_appropriate":       record.is_appropriate,
        "confidence":           record.confidence,
        "appropriateness_note": record.appropriateness_note,
    }

    record.is_appropriate       = decision
    record.appropriateness_note = note
    record.confidence           = confidence
    record.reviewed_by          = reviewed_by
    record.reviewed_at          = datetime.utcnow()

    _log_history(
        db, record.uuid, "reviewed",
        changed_by=reviewed_by,
        old_values=old_vals,
        new_values={"is_appropriate": decision, "confidence": confidence},
        reason=note,
    )

    db.commit()
    return True


def get_pending_review(
    db: Session,
    limit: int = 50,
    category: str = None,
) -> list[MedicalKnowledge]:
    query = db.query(MedicalKnowledge).filter(
        MedicalKnowledge.is_appropriate == "unknown"
    )
    if category:
        query = query.filter(MedicalKnowledge.category == category)
    return query.order_by(MedicalKnowledge.confidence.desc()).limit(limit).all()


def get_by_id(db: Session, knowledge_id: int) -> Optional[MedicalKnowledge]:
    return db.query(MedicalKnowledge).filter_by(id=knowledge_id).first()


def get_stats(db: Session) -> dict:
    total    = db.query(MedicalKnowledge).count()
    approved = db.query(MedicalKnowledge).filter_by(is_appropriate="yes").count()
    rejected = db.query(MedicalKnowledge).filter_by(is_appropriate="no").count()
    pending  = db.query(MedicalKnowledge).filter_by(is_appropriate="unknown").count()

    return {
        "total":          total,
        "approved":       approved,
        "rejected":       rejected,
        "pending_review": pending,
        "approval_rate":  round(approved / total * 100, 2) if total else 0,
    }


def log_extraction(
    db: Session,
    raw_text: str,
    extracted_json: dict,
    diagnosis_code: str = None,
    total_billed: float = None,
    confidence: float = None,
    model: str = "llama-3.1-70b-versatile",
) -> InvoiceExtractionLog:
    text_hash = hashlib.sha256((raw_text or "").encode()).hexdigest()[:64]

    log = InvoiceExtractionLog(
        ocr_text_hash         = text_hash,
        raw_ocr_text          = raw_text[:10000] if raw_text else None,
        extracted_json        = extracted_json,
        diagnosis_code        = diagnosis_code,
        total_billed          = total_billed,
        extraction_confidence = confidence,
        processed_by_model    = model,
    )
    db.add(log)
    db.commit()
    return log


def vector_search_sqlite(
    db: Session,
    query_text: str,
    diagnosis_code: str = None,
    top_k: int = 5,
    threshold: float = 0.70,
) -> list[dict]:
    import numpy as np

    if not db or not query_text:
        return []

    try:
        q_vec = _encode(query_text)
        if not q_vec:
            return []

        q_arr = np.array(q_vec, dtype="float32")

        query = db.query(MedicalKnowledge).filter(
            MedicalKnowledge.is_appropriate != "no",
            MedicalKnowledge.metadata_json.isnot(None),
        )
        if diagnosis_code:
            query = query.filter(MedicalKnowledge.diagnosis_code == diagnosis_code)

        records = query.limit(200).all()
        scored  = []

        for r in records:
            emb = (r.metadata_json or {}).get("embedding")
            if not emb:
                continue
            r_arr = np.array(emb, dtype="float32")
            sim   = float(np.dot(q_arr, r_arr))
            if sim >= threshold:
                scored.append((r, sim))

        scored.sort(key=lambda x: x[1], reverse=True)

        return [
            {
                "id":             r.id,
                "uuid":           r.uuid,
                "description":    r.description,
                "category":       r.category,
                "medical_code":   r.medical_code,
                "unit_price_avg": float(((r.unit_price_low or 0) + (r.unit_price_high or 0)) / 2),
                "unit_price_min": float(r.unit_price_low  or 0),
                "unit_price_max": float(r.unit_price_high or 0),
                "typical_total":  float(r.typical_total_low or 0),
                "sample_count":   (r.metadata_json or {}).get("sample_count", 1),
                "is_appropriate": r.is_appropriate or "unknown",
                "confidence":     float(r.confidence or 0),
                "similarity":     round(sim, 4),
            }
            for r, sim in scored[:top_k]
        ]

    except Exception as e:
        print(f"[knowledge_service] vector_search_sqlite error: {e}")
        return []


def _encode(text: str) -> list[float]:
    try:
        from .rag_service import encode
        return encode(text)
    except Exception:
        return []