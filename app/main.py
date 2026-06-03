import uuid
from fastapi import FastAPI, UploadFile, File, Depends, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel
from typing import Optional
from sqlalchemy.orm import Session
from sqlalchemy import text

from app.core.database_user      import get_db
from app.core.database_knowledge import get_knowledge_db
from app.models.models_user      import Claim, ClaimAuditLog
from app.schemas                 import ClaimInput
from app.services.claim_service  import process_claim as service_process_claim
from app.services.ocr_itemized   import process_ocr_claim, clean_ocr_text

app = FastAPI(title="AI Claim Approval System")


class ClaimRequest(BaseModel):
    text: str
    diagnosis_code: Optional[str] = None
    user_id: Optional[str] = None


def _audit(db, claim_id, action, user_id="system", details=None, request=None):
    ip         = request.client.host               if request else None
    user_agent = request.headers.get("user-agent") if request else None
    db.add(ClaimAuditLog(
        claim_id=claim_id, action=action, user_id=user_id,
        ip_address=ip, user_agent=user_agent, details=details,
    ))
    db.commit()


@app.exception_handler(RequestValidationError)
async def _val_err(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=200, content={
        "status": "invalid_input", "decision": "Pending",
        "confidence": 0.0, "reason": "validation_failed", "errors": exc.errors(),
    })


@app.get("/")
def home():
    return {"status": "ok", "message": "AI Claim Approval System is running"}


@app.get("/health")
def health(db: Session = Depends(get_db), db_knowledge: Session = Depends(get_knowledge_db)):
    results = {}
    for label, sess in [("claims_db", db), ("knowledge_db", db_knowledge)]:
        try:
            sess.execute(text("SELECT 1"))
            results[label] = "connected"
        except Exception as e:
            results[label] = f"error: {e}"
    return {"status": "ok" if all(v == "connected" for v in results.values()) else "degraded", **results}


@app.post("/predict")
def predict(data: ClaimInput, request: Request,
            db: Session = Depends(get_db), db_knowledge: Session = Depends(get_knowledge_db)):
    try:
        result   = service_process_claim(form_data=data.model_dump(), db_knowledge=db_knowledge, db_claims=db)
        claim_id = result.get("claim_id") or f"CLM-{uuid.uuid4().hex[:8].upper()}"
        _audit(db, claim_id, "PREDICT", details=f"Decision={result.get('decision')}", request=request)
        return {"status": "ok", "claim_id": claim_id, **result}
    except Exception as e:
        return {"status": "system_error", "decision": "Pending", "reason": str(e)}


@app.post("/ocr-claim")
def ocr_claim(file: UploadFile = File(...), request: Request = None,
              db: Session = Depends(get_db), db_knowledge: Session = Depends(get_knowledge_db)):
    try:
        result   = service_process_claim(file_bytes=file.file.read(), db_knowledge=db_knowledge, db_claims=db)
        claim_id = result.get("claim_id") or "unknown"
        _audit(db, claim_id, "OCR_CLAIM",
               details=f"Items={result.get('items_count', 0)}, Decision={result.get('decision')}",
               request=request)
        return result
    except Exception as e:
        return {"status": "ocr_failed", "error": str(e)}


@app.post("/ocr-claim/form")
async def ocr_claim_form(file: UploadFile = File(...), request: Request = None,
                         db: Session = Depends(get_db), db_knowledge: Session = Depends(get_knowledge_db)):
    try:
        form      = await request.form()
        form_data = {k: v for k, v in form.items() if k != "file" and v is not None}
        result    = service_process_claim(
            file_bytes=await file.read(),
            form_data=form_data or None,
            db_knowledge=db_knowledge,
            db_claims=db,
        )
        return result
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/ocr-claim/text")
def ocr_claim_text(req: ClaimRequest, request: Request,
                   db: Session = Depends(get_db)):
    try:
        user_id        = req.user_id or "default_user"
        existing_bills = [
            c.extracted_data for c in
            db.query(Claim).filter(Claim.user_id == user_id).order_by(Claim.created_at.desc()).limit(50).all()
            if c.extracted_data
        ]
        result = process_ocr_claim(
            text=req.text,
            diagnosis_code=req.diagnosis_code,
            existing_bills=existing_bills,
            user_id=user_id,
        )
        print(f"[ocr-claim/text] status={result['status']} duplicate={result['duplicate']}")
        if result["status"] == "ok":
            bill     = result["bill"]
            claim_id = f"CLM-{uuid.uuid4().hex[:8].upper()}"
            db.add(Claim(
                claim_id=claim_id,
                user_id=user_id,
                extracted_data=bill,
                total_billed=bill.get("total_billed"),
                currency=bill.get("currency"),
                patient_name=bill.get("patient_name"),
                diagnosis_text=bill.get("diagnosis_text"),
                bill_date=bill.get("bill_date"),
            ))
            db.commit()
            _audit(db, claim_id, "OCR_TEXT_SAVED",
                   details=f"Items={len(bill.get('items', []))}, Total={bill.get('total_billed')}",
                   request=request)
            print(f"[ocr-claim/text] Bill SAVED claim_id={claim_id}")
        elif result["status"] == "duplicate":
            print(f"[ocr-claim/text] DUPLICATE reason={result.get('reason')}")
        return {
            "status":            result["status"],
            "duplicate":         result["duplicate"],
            "reason":            result.get("reason"),
            "extracted_data":    result.get("bill"),
            "total_bills_in_db": db.query(Claim).filter(Claim.user_id == user_id).count(),
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/claims")
def get_claims(limit: int = 50, db: Session = Depends(get_db)):
    return db.query(Claim).order_by(Claim.created_at.desc()).limit(limit).all()


@app.get("/claims/{claim_id}")
def get_claim(claim_id: str, db: Session = Depends(get_db)):
    c = db.query(Claim).filter(Claim.claim_id == claim_id).first()
    return c if c else {"status": "not_found"}


@app.get("/knowledge/stats")
def knowledge_stats(db_knowledge: Session = Depends(get_knowledge_db)):
    from app.services.knowledge_service import get_stats
    return get_stats(db_knowledge)


@app.get("/knowledge/pending")
def knowledge_pending(limit: int = 50, category: str = None,
                      db_knowledge: Session = Depends(get_knowledge_db)):
    from app.services.knowledge_service import get_pending_review
    records = get_pending_review(db_knowledge, limit=limit, category=category)
    return [{
        "id":              r.id,
        "uuid":            r.uuid,
        "description":     r.description,
        "category":        r.category,
        "diagnosis_code":  r.diagnosis_code,
        "confidence":      r.confidence,
        "sample_count":    (r.metadata_json or {}).get("sample_count", 1),
        "unit_price_low":  r.unit_price_low,
        "unit_price_high": r.unit_price_high,
        "created_at":      str(r.created_at),
    } for r in records]


@app.post("/knowledge/{knowledge_id}/review")
def review_knowledge(knowledge_id: int, decision: str, note: str = None,
                     reviewed_by: str = "expert",
                     db_knowledge: Session = Depends(get_knowledge_db)):
    from app.services.knowledge_service import expert_review
    ok = expert_review(db_knowledge, knowledge_id, decision, note, reviewed_by=reviewed_by)
    return {"status": "ok" if ok else "error", "knowledge_id": knowledge_id, "decision": decision}