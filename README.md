# AI-Powered Health Insurance Claim Decision System  
## (RAG + OCR + Knowledge Base + Fraud Detection + Explainability + Audit Logging)

---

## 1. Project Overview

This system automates health insurance claim adjudication using a hybrid AI architecture combining structured validation, OCR extraction, Retrieval-Augmented Generation (RAG), explainable AI techniques, and a secure knowledge + audit system.

The system is designed to ensure that all claim decisions are:

- grounded in retrieved evidence  
- explainable at both reasoning and feature level  
- auditable  
- resistant to hallucination  
- robust to fraudulent and anomalous claims  

It also supports reimbursement estimation and fraud risk detection.

---
## 2. Core Features

- **AI Decision Engine**: Automatic classification (Approved/Denied/Pending), reimbursement calculation, confidence scoring, and clear explanations.
- **RAG Retrieval System**: FAISS-powered semantic search to retrieve relevant historical claims, policies, and medical rules for grounded reasoning.
- **Advanced OCR Pipeline**: Extracts structured data from medical invoices with robust validation and error correction.
- **Knowledge Base**: Centralized storage of insurance policies, pricing rules, treatment guidelines, and fraud patterns.
- **Fraud Detection**: Hybrid system combining rule-based checks and machine learning to identify suspicious claims and overbilling.
- **Explainability**: SHAP-based feature attribution for transparent, interpretable decisions.
- **Audit Logging**: Complete tracking of all actions for compliance, traceability, and system monitoring.

---

## 3. System Architecture

The system is built on a modular, multi-layer architecture:

- **Input Layer**: Web form and medical invoice image upload
- **Processing Layer**: Data validation, normalization, and OCR extraction
- **Retrieval Layer (RAG)**: FAISS semantic search for relevant context
- **Intelligence Layer**: Groq LLM (LLaMA 3.1) for evidence-based reasoning
- **Fraud Detection Layer**: Real-time risk assessment and anomaly detection
- **Explainability Layer**: SHAP analysis for feature-level insights
- **Persistence Layer**: Claims database, Knowledge base, and immutable Audit logs

---

## 4. Workflow

1. User submits claim through form or uploads invoice images
2. System validates and normalizes input data
3. OCR extracts structured information from images (if provided)
4. RAG retrieves relevant policies, similar claims, and guidelines
5. Fraud Detection Layer analyzes risk and flags anomalies
6. LLM generates a reasoned decision based on retrieved evidence
7. SHAP explains which factors influenced the final decision
8. System returns decision, reimbursement amount, explanation, and fraud risk score
9. All actions are securely logged in the audit database

---

## 5. Databases

- **Claims Database** (`health_claims.db`): Stores all submitted claims, decisions, reimbursement details, and metadata.
- **Knowledge Database** (`medical_knowledge.db`): Contains insurance rules, pricing data, medical guidelines, fraud patterns, and embeddings for RAG.
- **Audit Database**: Immutable logs of every system action, ensuring full traceability and regulatory compliance.

---

## 6. Tech Stack

Backend: FastAPI  
- LLM: Groq (LLaMA 3.1)  
- Embeddings: Sentence Transformers (all-MiniLM-L6-v2)  
- Vector Search: FAISS  
- OCR: Tesseract / PaddleOCR  
- Explainability: SHAP  
- Fraud Detection: Machine Learning + rule-based hybrid  
- Frontend: Streamlit  
- Storage: SQLite (separated Knowledge DB + Audit DB)

---

## 7. How to Run

### 7.1 Install Dependencies
pip install -r requirements.txt

### 7.2 Run Backend (FastAPI)
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

### 7.3 Run Frontend (Streamlit)
streamlit run frontend.py

---

## 8. Key Design Principles

- No free-text reasoning without retrieval grounding  
- Strict schema validation before LLM processing  
- Separation of knowledge, audit, reasoning, and explainability layers  
- Hybrid fraud detection (ML + rules)  
- Feature-level and reasoning-level explainability (SHAP + LLM)  
- Full traceability of system decisions  
- Fraud-aware decision pipeline by design  

---

## 9. Summary

This system is a production-oriented AI decision engine for insurance claim processing. It integrates OCR, RAG, fraud detection, SHAP-based explainability, and audit logging to ensure decisions are grounded, interpretable at multiple levels, and fully traceable in real-world deployment scenarios.