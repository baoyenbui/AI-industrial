import uuid
from fastapi import FastAPI, UploadFile, File, Depends, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from sqlalchemy.orm import Session
from sqlalchemy import text

from app.core.database_user      import get_db
from app.core.database_knowledge import get_knowledge_db
from app.models.models_user      import Claim, ClaimAuditLog
from app.schemas                 import ClaimInput
from app.services.claim_service  import process_claim

app = FastAPI(title="AI Claim Approval System")

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
        result   = process_claim(form_data=data.model_dump(), db_knowledge=db_knowledge, db_claims=db)
        claim_id = result.get("claim_id") or f"CLM-{uuid.uuid4().hex[:8].upper()}"
        _audit(db, claim_id, "PREDICT", details=f"Decision={result.get('decision')}", request=request)
        return {"status": "ok", "claim_id": claim_id, **result}
    except Exception as e:
        return {"status": "system_error", "decision": "Pending", "reason": str(e)}


@app.post("/ocr-claim")
def ocr_claim(file: UploadFile = File(...), request: Request = None,
              db: Session = Depends(get_db), db_knowledge: Session = Depends(get_knowledge_db)):
    try:
        result   = process_claim(file_bytes=file.file.read(), db_knowledge=db_knowledge, db_claims=db)
        claim_id = result.get("claim_id") or "unknown"
        _audit(db, claim_id, "OCR_CLAIM",
               details=f"Items={result.get('items_count',0)}, Decision={result.get('decision')}",
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
        result    = process_claim(file_bytes=await file.read(), form_data=form_data or None,
                                  db_knowledge=db_knowledge, db_claims=db)
        return result
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
        "id": r.id, "uuid": r.uuid, "description": r.description,
        "category": r.category, "diagnosis_code": r.diagnosis_code,
        "confidence": r.confidence,
        "sample_count": (r.metadata_json or {}).get("sample_count", 1),
        "unit_price_low": r.unit_price_low, "unit_price_high": r.unit_price_high,
        "created_at": str(r.created_at),
    } for r in records]


@app.post("/knowledge/{knowledge_id}/review")
def review_knowledge(knowledge_id: int, decision: str, note: str = None,
                     reviewed_by: str = "expert",
                     db_knowledge: Session = Depends(get_knowledge_db)):
    from app.services.knowledge_service import expert_review
    ok = expert_review(db_knowledge, knowledge_id, decision, note, reviewed_by=reviewed_by)
    return {"status": "ok" if ok else "error", "knowledge_id": knowledge_id, "decision": decision}