# AI-Powered Health Insurance Claim Approval System

## 1. Project Overview
This system automates health insurance claim adjudication using a hybrid AI pipeline combining structured validation, OCR extraction, FAISS-based retrieval, and LLM reasoning (Groq LLaMA 3.1).

It supports both manual structured input and invoice image upload, ensuring all claims are validated before AI processing.

---

## 2. Features
- AI-assisted claim decision (Approved / Denied / Pending)
- Retrieval-Augmented Generation (FAISS similar claim search)
- Strict schema-based validation (no free-text ambiguity)
- OCR-based invoice extraction (image → structured data)
- Automatic rejection of incomplete or invalid claims
- Explainable AI decisions with retrieved evidence
- Streamlit interactive dashboard

---

## 3. Workflow
1. User submits claim (form or invoice image)
2. System validates and normalizes input
3. OCR extracts fields if image is uploaded
4. FAISS retrieves similar historical claims
5. LLM generates decision based only on retrieved evidence
6. System returns structured output (decision, reason, confidence)

---

## 4. Tech Stack
Backend: FastAPI  
LLM: Groq (LLaMA 3.1)  
Embeddings: Sentence Transformers (all-MiniLM-L6-v2)  
Vector Search: FAISS  
OCR: Tesseract / PaddleOCR  
Frontend: Streamlit  

---

## 5. System Architecture
- Input Layer: Structured form + optional OCR image input  
- Processing Layer: Validation + normalization + OCR extraction  
- Retrieval Layer: FAISS similarity search (top-K claims)  
- Intelligence Layer: LLM reasoning using retrieved cases  
- Output Layer: Decision + explanation + confidence score  

---

## 6. Installation & Run

### 6.1 Install Dependencies
pip install fastapi uvicorn sentence-transformers faiss-cpu pandas numpy streamlit groq pytesseract

### 6.2 Run Backend
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

### 6.3 Run Frontend
streamlit run frontend.py

---

## 7. Key Improvements
- Prevents incomplete claims from reaching LLM
- Enforces strict structured input pipeline
- Adds OCR support for real-world usage
- Improves retrieval quality via FAISS
- Ensures explainable and auditable AI decisions