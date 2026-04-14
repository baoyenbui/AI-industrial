import streamlit as st
import requests
from PIL import Image
import re

API_URL = "http://127.0.0.1:8000/predict"
OCR_API_URL = "http://127.0.0.1:8000/ocr-claim"

st.set_page_config(page_title="Health Insurance Claim AI", layout="wide")

def safe_str(x):
    if x is None:
        return "None"
    x = str(x).strip()
    return x if x else "None"

def safe_int(x, default=0):
    try:
        if x is None:
            return default
        if isinstance(x, str):
            m = re.search(r"\d+", x)
            return int(m.group()) if m else default
        return int(x)
    except:
        return default

def safe_float(x, default=0.0):
    try:
        if x is None:
            return default
        return float(str(x).replace(",", "."))
    except:
        return default

def safe_gender(x):
    if not x:
        return "None"
    x = str(x).lower()
    if "m" in x:
        return "Male"
    if "f" in x:
        return "Female"
    return "None"

if "patient_age" not in st.session_state:
    st.session_state.patient_age = 0
if "patient_gender" not in st.session_state:
    st.session_state.patient_gender = "None"
if "employee_status" not in st.session_state:
    st.session_state.employee_status = "None"
if "marital_status" not in st.session_state:
    st.session_state.marital_status = "None"
if "diagnosis" not in st.session_state:
    st.session_state.diagnosis = "None"
if "procedure" not in st.session_state:
    st.session_state.procedure = "None"
if "claim_amount" not in st.session_state:
    st.session_state.claim_amount = 0.0

st.title("Health Insurance Claim Approval AI")

uploaded_file = st.file_uploader("Upload claim image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=300)

    if st.button("Extract Information from Image"):
        try:
            uploaded_file.seek(0)

            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    uploaded_file.type
                )
            }

            response = requests.post(OCR_API_URL, files=files)

            if response.status_code == 200:
                data = response.json()
                extracted = data.get("extracted_data") or {}

                st.success("OCR Done")

                st.session_state.patient_age = safe_int(extracted.get("PatientAge"), 0)
                st.session_state.patient_gender = safe_gender(extracted.get("PatientGender"))
                st.session_state.employee_status = safe_str(extracted.get("PatientEmploymentStatus"))
                st.session_state.diagnosis = safe_str(extracted.get("DiagnosisCode"))
                st.session_state.procedure = safe_str(extracted.get("ProcedureCode"))
                st.session_state.claim_amount = safe_float(extracted.get("ClaimAmount"), 0.0)

                st.expander("Raw OCR Text").write(data.get("raw_text", ""))

                st.expander("Extracted Data").json({
                    "PatientAge": st.session_state.patient_age,
                    "PatientGender": st.session_state.patient_gender,
                    "PatientEmploymentStatus": st.session_state.employee_status,
                    "DiagnosisCode": st.session_state.diagnosis,
                    "ProcedureCode": st.session_state.procedure,
                    "ClaimAmount": st.session_state.claim_amount
                })

            else:
                st.error("OCR failed")

        except Exception as e:
            st.error(str(e))

with st.form("claim_form"):
    patient_age = st.number_input(
        "Patient Age",
        min_value=0,
        max_value=120,
        value=st.session_state.patient_age,
        key="patient_age"
    )

    patient_gender = st.selectbox(
        "Gender",
        ["Male", "Female", "None"],
        index=["Male", "Female", "None"].index(st.session_state.patient_gender)
    )

    employee_status = st.text_input("Employment Status", st.session_state.employee_status)
    marital_status = st.text_input("Marital Status", st.session_state.marital_status)
    diagnosis = st.text_input("Diagnosis Code", st.session_state.diagnosis)
    procedure = st.text_input("Procedure Code", st.session_state.procedure)

    claim_amount = st.number_input(
        "Claim Amount",
        min_value=0.0,
        value=st.session_state.claim_amount
    )

    submitted = st.form_submit_button("Submit")

if submitted:
    query = f"Patient {patient_gender}, {patient_age} years old, {employee_status}, {marital_status}, Diagnosis {diagnosis}, Procedure {procedure}, Amount {claim_amount}"

    try:
        res = requests.get(API_URL, params={"query": query}, timeout=30)

        if res.status_code == 200:
            st.success("Decision received")
            st.json(res.json())
        else:
            st.error("API error")

    except Exception as e:
        st.error(str(e))