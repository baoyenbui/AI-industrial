import json
import uuid
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, Depends, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from sqlalchemy.orm import Session
from sqlalchemy import text

from app.core.database_user      import get_db
from app.core.database_knowledge import get_knowledge_db
from app.models.models_user      import Claim, ClaimAuditLog
from app.schemas                 import ClaimInput

from app.services.claim_service  import process_claim, get_decision, _build_data, _save_claim
from app.services.extraction_service import parse_query

app = FastAPI(title="AI Claim Approval System")


def _audit(
    db: Session,
    claim_id: str,
    action: str,
    user_id: str = "system",
    details: str = None,
    request: Request = None,
):
    ip         = request.client.host                  if request else None
    user_agent = request.headers.get("user-agent")   if request else None
    db.add(ClaimAuditLog(
        claim_id=claim_id, action=action, user_id=user_id,
        ip_address=ip, user_agent=user_agent, details=details,
    ))
    db.commit()


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=200,
        content={
            "status":    "invalid_input",
            "decision":  "Pending",
            "confidence": 0.0,
            "reason":    "validation_failed",
            "errors":    exc.errors(),
        },
    )
@app.get("/")
def home():
    return {"status": "ok", "message": "AI Claim Approval System is running"}


@app.get("/health")
def health_check(
    db:           Session = Depends(get_db),
    db_knowledge: Session = Depends(get_knowledge_db),
):
    results = {}
    for label, session in [("claims_db", db), ("knowledge_db", db_knowledge)]:
        try:
            session.execute(text("SELECT 1"))
            results[label] = "connected"
        except Exception as e:
            results[label] = f"error: {e}"

    overall = "ok" if all("connected" == v for v in results.values()) else "degraded"
    return {"status": overall, **results}


@app.post("/predict")
def predict(
    data:         ClaimInput,
    request:      Request,
    db:           Session = Depends(get_db),
    db_knowledge: Session = Depends(get_knowledge_db),
):
    try:
        clean_data = data.model_dump()

        parsed = parse_query(clean_data)

        from app.services.rag_service import vector_search, build_rag_context
        diagnosis_code = parsed.get("diagnosiscode") or ""
        similar        = vector_search(db_knowledge, str(clean_data), diagnosis_code) if db_knowledge else []
        rag_context    = build_rag_context([], similar)

        decision_result = get_decision(parsed, rag_context, itemized_items=[])

        # Lấy data đã build trong get_decision
        from app.utils import safe_float
        data_for_save = decision_result.pop("_data", None)
        if data_for_save is None:
            claim_amount  = safe_float(parsed.get("claimamount") or parsed.get("amount") or 0)
            data_for_save = _build_data(parsed, claim_amount)

        # Save qua _save_claim — không duplicate logic ở đây
        _save_claim(db, data_for_save, decision_result, raw_ocr_text="")

        # Lấy claim_id vừa tạo để ghi audit (query record mới nhất của session này)
        saved = db.query(Claim).order_by(Claim.id.desc()).first()
        claim_id = saved.claim_id if saved else f"CLM-{uuid.uuid4().hex[:8].upper()}"

        _audit(
            db, claim_id, "PREDICT", "system",
            f"Decision={decision_result.get('decision')}, "
            f"Reimbursement={decision_result.get('reimbursement_amount')}",
            request,
        )

        return {"status": "ok", "claim_id": claim_id, **decision_result}

    except Exception as e:
        print(f"[/predict] ERROR: {e}")
        return {"status": "system_error", "decision": "Pending", "confidence": 0.0, "reason": str(e)}


@app.post("/ocr-claim")
def ocr_claim(
    file:         UploadFile = File(...),
    request:      Request    = None,
    db:           Session    = Depends(get_db),
    db_knowledge: Session    = Depends(get_knowledge_db),
):
    try:
        result = process_claim(
            file_bytes   = file.file.read(),
            db_knowledge = db_knowledge,
            db_claims    = db,
        )

        claim_id = result.get("claim_id") or "unknown"

        # Tìm claim vừa save để lấy claim_id thực
        saved = db.query(Claim).order_by(Claim.id.desc()).first()
        if saved:
            claim_id = saved.claim_id

        _audit(
            db, claim_id, "OCR_CLAIM", "system",
            f"Items={result.get('items_count', 0)}, "
            f"Decision={result.get('decision')}",
            request,
        )

        return result

    except Exception as e:
        print(f"[/ocr-claim] ERROR: {e}")
        return {"status": "ocr_failed", "error": str(e)}


@app.post("/ocr-claim/form")
async def ocr_claim_with_form(
    file:         UploadFile  = File(...),
    request:      Request     = None,
    db:           Session     = Depends(get_db),
    db_knowledge: Session     = Depends(get_knowledge_db),
):
    """OCR file rồi merge với form fields từ request body nếu có."""
    try:
        form       = await request.form()
        form_data  = {k: v for k, v in form.items() if k != "file"}

        result = process_claim(
            file_bytes   = await file.read(),
            form_data    = form_data or None,
            db_knowledge = db_knowledge,
            db_claims    = db,
        )
        return result

    except Exception as e:
        print(f"[/ocr-claim/form] ERROR: {e}")
        return {"status": "ocr_failed", "error": str(e)}

@app.get("/claims")
def get_claims(
    limit: int     = 50,
    db:    Session = Depends(get_db),
):
    claims = (
        db.query(Claim)
        .order_by(Claim.created_at.desc())
        .limit(limit)
        .all()
    )
    return claims


@app.get("/claims/{claim_id}")
def get_claim(claim_id: str, request: Request, db: Session = Depends(get_db)):
    claim = db.query(Claim).filter(Claim.claim_id == claim_id).first()
    if not claim:
        return {"status": "not_found", "message": "Claim not found"}

    _audit(db, claim_id, "VIEW", "system", "Viewed claim details", request)
    return claim


@app.get("/audit-logs")
def get_audit_logs(limit: int = 100, db: Session = Depends(get_db)):
    return (
        db.query(ClaimAuditLog)
        .order_by(ClaimAuditLog.created_at.desc())
        .limit(limit)
        .all()
    )


@app.get("/knowledge/stats")
def knowledge_stats(db_knowledge: Session = Depends(get_knowledge_db)):
    from app.services.knowledge_service import get_stats
    return get_stats(db_knowledge)


@app.get("/knowledge/pending")
def knowledge_pending(
    limit:    int    = 50,
    category: str    = None,
    db_knowledge: Session = Depends(get_knowledge_db),
):
    from app.services.knowledge_service import get_pending_review
    records = get_pending_review(db_knowledge, limit=limit, category=category)
    return [
        {
            "id":             r.id,
            "uuid":           r.uuid,
            "description":    r.description,
            "category":       r.category,
            "diagnosis_code": r.diagnosis_code,
            "confidence":     r.confidence,
            "sample_count":   (r.metadata_json or {}).get("sample_count", 1),
            "unit_price_low": r.unit_price_low,
            "unit_price_high":r.unit_price_high,
            "created_at":     str(r.created_at),
        }
        for r in records
    ]


@app.post("/knowledge/{knowledge_id}/review")
def review_knowledge(
    knowledge_id: int,
    decision:     str,
    note:         str   = None,
    reviewed_by:  str   = "expert",
    db_knowledge: Session = Depends(get_knowledge_db),
):
    from app.services.knowledge_service import expert_review
    ok = expert_review(db_knowledge, knowledge_id, decision, note, reviewed_by=reviewed_by)
    if not ok:
        return {"status": "error", "message": "Record not found or invalid decision"}
    return {"status": "ok", "knowledge_id": knowledge_id, "decision": decision}