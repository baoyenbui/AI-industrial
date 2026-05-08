import os
import json
import numpy as np
from typing import Optional
from sentence_transformers import SentenceTransformer
from sqlalchemy.orm import Session
from dotenv import load_dotenv

load_dotenv()

_encoder: Optional[SentenceTransformer] = None

def _get_encoder() -> SentenceTransformer:
    global _encoder
    if _encoder is None:
        _encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _encoder


def encode(text: str) -> list[float]:
    if not text or not text.strip():
        return []
    encoder = _get_encoder()
    vec = encoder.encode(text.strip(), normalize_embeddings=True)
    return vec.tolist()


def vector_search(
    db: Session,
    query_text: str,
    diagnosis_code: str = None,
    top_k: int = 5,
    threshold: float = 0.70
) -> list[dict]:
    """
    Vector search dùng numpy cosine similarity — tương thích SQLite.
    Không dùng pgvector extension.
    Embedding được lưu trong metadata_json["embedding"].
    """
    if not db or not query_text or not query_text.strip():
        return []

    try:
        from ..models.models_knowledge import MedicalKnowledge

        q_vec = encode(query_text)
        if not q_vec:
            return []

        q_arr = np.array(q_vec, dtype="float32")

        # Lấy toàn bộ record chưa bị reject để tính cosine thủ công
        # MedicalKnowledge không có column review_status → dùng is_appropriate
        base_query = db.query(MedicalKnowledge).filter(
            MedicalKnowledge.is_appropriate != "no"
        )

        if diagnosis_code and diagnosis_code.strip():
            base_query = base_query.filter(
                MedicalKnowledge.diagnosis_code == diagnosis_code.strip()
            )

        records = base_query.all()

        scored = []
        for r in records:
            # Embedding lưu trong metadata_json["embedding"]
            meta = r.metadata_json or {}
            emb = meta.get("embedding")

            if not emb:
                continue

            r_arr = np.array(emb, dtype="float32")

            # Cả 2 vec đã normalize khi encode → dot product = cosine similarity
            if r_arr.shape != q_arr.shape:
                continue

            similarity = float(np.dot(q_arr, r_arr))

            if similarity >= threshold:
                scored.append((r, similarity))

        # Sort descending, lấy top_k
        scored.sort(key=lambda x: x[1], reverse=True)
        scored = scored[:top_k]

        return [
            {
                "id":             str(r.id),
                "description":    r.description,
                "category":       r.category,
                "medical_code":   r.medical_code,
                # unit_price_avg = midpoint of low/high
                "unit_price_avg": float(((r.unit_price_low or 0) + (r.unit_price_high or 0)) / 2),
                "unit_price_min": float(r.unit_price_low  or 0),
                "unit_price_max": float(r.unit_price_high or 0),
                "typical_total":  float(r.typical_total_low or 0),
                # sample_count lưu trong metadata_json
                "sample_count":   (r.metadata_json or {}).get("sample_count", 1),
                "is_appropriate": r.is_appropriate or "unknown",
                # MedicalKnowledge không có review_status → dùng is_appropriate làm proxy
                "review_status":  r.is_appropriate or "unknown",
                "confidence":     float(r.confidence or 0),
                "similarity":     round(sim, 4),
            }
            for r, sim in scored
        ]

    except Exception as e:
        print(f"[rag_service] vector_search error: {e}")
        return []


def build_rag_context(
    claim_items: list[dict],
    similar_knowledge: list[dict]
) -> str:
    if not similar_knowledge:
        return "No similar cases found in knowledge base."

    lines = [
        f"KNOWLEDGE BASE — {len(similar_knowledge)} similar record(s) found:",
        ""
    ]

    for k in similar_knowledge:
        status_icon = (
            "APPROPRIATE" if k["is_appropriate"] == "yes" else
            "FLAGGED"     if k["is_appropriate"] == "no"  else
            "UNREVIEWED"
        )

        price_range = ""
        if k["unit_price_min"] and k["unit_price_max"]:
            price_range = f" (range ${k['unit_price_min']:,.2f}–${k['unit_price_max']:,.2f})"

        lines.append(
            f"• [{status_icon}] {k['description']}"
            f" | avg ${k['unit_price_avg']:,.2f}{price_range}"
            f" | seen {k['sample_count']}x"
            f" | similarity {k['similarity']:.0%}"
            + (f" | code {k['medical_code']}" if k.get("medical_code") else "")
        )

    if claim_items:
        lines += ["", "CLAIM ITEMS vs KNOWLEDGE BASE:"]
        for item in claim_items:
            desc  = item.get("description", "")
            total = item.get("total") or 0
            best  = _find_best_match(desc, similar_knowledge)

            if best:
                avg      = best["unit_price_avg"]
                qty      = item.get("quantity") or 1
                expected = avg * qty
                diff_pct = ((total - expected) / expected * 100) if expected else 0
                flag     = " ⚠ OVERPRICED" if diff_pct > 30 else ""
                lines.append(
                    f"  - {desc}: billed ${total:,.2f}"
                    f" | expected ~${expected:,.2f}{flag}"
                )
            else:
                lines.append(f"  - {desc}: billed ${total:,.2f} | no reference found")

    return "\n".join(lines)


def _find_best_match(
    description: str,
    knowledge_list: list[dict],
    min_similarity: float = 0.75
) -> Optional[dict]:
    if not description or not knowledge_list:
        return None

    best_score  = -1.0
    best_record = None

    for k in knowledge_list:
        if k.get("similarity", 0) >= min_similarity:
            if k["similarity"] > best_score:
                best_score  = k["similarity"]
                best_record = k

    return best_record


def rerank_by_diagnosis(
    results: list[dict],
    diagnosis_code: str,
    boost: float = 0.05
) -> list[dict]:
    if not diagnosis_code:
        return results

    for r in results:
        if r.get("diagnosis_code") == diagnosis_code:
            r["similarity"] = min(1.0, r["similarity"] + boost)

    return sorted(results, key=lambda x: x["similarity"], reverse=True)


def embed_and_store(
    db: Session,
    knowledge_id: str,
    description: str
) -> bool:
    """
    Lưu embedding vào metadata_json["embedding"] — không dùng column riêng.
    """
    if not db or not knowledge_id or not description:
        return False

    try:
        from ..models.models_knowledge import MedicalKnowledge

        record = db.query(MedicalKnowledge).filter_by(id=knowledge_id).first()
        if not record:
            return False

        emb = encode(description)
        if not emb:
            print(f"[rag_service] embed_and_store: empty embedding for id={knowledge_id}")
            return False

        meta = record.metadata_json or {}
        meta["embedding"] = emb
        record.metadata_json = meta

        db.commit()
        return True

    except Exception as e:
        print(f"[rag_service] embed_and_store error: {e}")
        db.rollback()
        return False


def batch_embed_missing(db: Session, batch_size: int = 100) -> int:
    """
    Backfill embedding cho các record thiếu — đọc/ghi qua metadata_json["embedding"].
    """
    if not db:
        return 0

    try:
        from ..models.models_knowledge import MedicalKnowledge

        all_records = db.query(MedicalKnowledge).all()

        # Lọc record chưa có embedding trong metadata_json
        missing = [
            r for r in all_records
            if not (r.metadata_json or {}).get("embedding")
        ][:batch_size]

        if not missing:
            return 0

        texts   = [r.description for r in missing]
        encoder = _get_encoder()
        vecs    = encoder.encode(texts, normalize_embeddings=True, batch_size=32)

        for record, vec in zip(missing, vecs):
            meta = record.metadata_json or {}
            meta["embedding"] = vec.tolist()
            record.metadata_json = meta

        db.commit()
        return len(missing)

    except Exception as e:
        print(f"[rag_service] batch_embed_missing error: {e}")
        db.rollback()
        return 0