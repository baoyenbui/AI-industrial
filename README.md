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

### AI Decision Engine
- Claim classification: Approved / Denied / Pending  
- Reimbursement amount prediction  
- Confidence scoring  
- Explanation grounded in retrieved evidence  
- Feature-level explainability using SHAP  

### RAG-Based Retrieval System
- FAISS-based similarity search  
- Retrieval of historical claims  
- Retrieval of insurance policies and medical rules  
- Context-aware LLM reasoning  

### OCR Pipeline
- Invoice image to structured data extraction  
- Robust parsing with validation layer  
- Handles noisy real-world documents  

### Knowledge Base System
- Insurance policies  
- Pricing rules  
- Medical guidelines  
- Fraud detection patterns  

### Fraud Detection Layer
- ML-based anomaly detection for fraud scoring  
- Rule-based anomaly detection  
- Suspicious prescription patterns  
- Overbilling detection signals  
- Risk scoring integration into decision process  

### Explainability Layer
- SHAP-based feature attribution for model decisions  
- Global + local feature importance analysis  
- Supports transparency in reimbursement and fraud scoring  

### Audit Logging System
- Records all system actions  
- Tracks claim and knowledge updates  
- Ensures traceability and compliance  
- Supports debugging and monitoring  

---

## 3. System Architecture

The system follows a multi-layer pipeline architecture:

### Input Layer
- Structured form input  
- Invoice image input  

### Processing Layer
- Input validation  
- Data normalization  
- OCR extraction (if image provided)  

### Retrieval Layer (RAG)
- FAISS similarity search  
- Retrieves:
  - similar claims  
  - policy rules  
  - fraud patterns  

### Intelligence Layer
- LLM reasoning (Groq LLaMA 3.1)  
- Decision generation based only on retrieved context  

### Explainability Layer
- SHAP-based feature attribution  
- Feature-level contribution analysis for:
  - approval decision  
  - reimbursement amount  
  - fraud risk score  

### Fraud Detection Layer
- ML-based anomaly detection model  
- Rule-based fraud signals  
- Integrated risk scoring into decision pipeline  

### Knowledge Layer
- Versioned knowledge base  
- Embedding storage for retrieval  
- Used for reasoning and fraud detection  

### Audit Layer
- Immutable audit logs  
- Tracks all actions on claims and knowledge updates   

---

## 4. Workflow

1. User submits claim (form or invoice image)  
2. System validates and normalizes input  
3. OCR extracts structured data if image is provided  
4. RAG retrieves relevant context:
   - historical claims  
   - insurance policies  
   - fraud patterns  
5. Fraud detection layer evaluates anomaly risk using ML model  
6. LLM generates decision using retrieved context only  
7. SHAP explains feature-level impact of decision  
8. System outputs:
   - decision  
   - reimbursement amount  
   - explanation  
   - fraud risk score  
   - SHAP feature attribution  
9. All actions are recorded in audit database  

---

## 5. Databases

### Knowledge Database

Stores domain intelligence used by the system:
- insurance policies  
- pricing rules  
- fraud patterns  
- embedding vectors for retrieval  

Purpose: supports RAG reasoning, enables fraud detection and drives reimbursement logic  

---

### Audit Database

Stores all system activity:
- claim lifecycle changes  
- knowledge updates  
- system/user actions  

Purpose: security and compliance, full traceability, debugging and monitoring  

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

## 7.How to Run

```bash

- pip install -r requirements.txt   # install dependencies
- uvicorn app.main:app --reload    # run backend API
- streamlit run app/frontend.py    # run UI

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