import streamlit as st
import requests
from PIL import Image
import io
import re
import mimetypes

API_URL = "http://127.0.0.1:8000/predict"
OCR_API_URL = "http://127.0.0.1:8000/ocr-claim"

st.set_page_config(page_title="Health Claim AI", layout="wide")

st.markdown("""
<style>

.stApp {
    background-color: white !important;
    border: none;
    box-shadow: 0 0 0 6px rgba(0, 0, 0, 0.25);
}

* {
    color: black !important;
}

section, div {
    background-color: transparent !important;
}

.block-container {
    background-color: white !important;
}

div[data-testid="stFileUploader"] {
    background-color: #E6F0FF !important;
    border-radius: 12px;
    padding: 12px;
}

div[data-testid="stFileUploaderDropzone"],
div[data-testid="stFileUploaderDropzoneInstructions"] {
    background-color: #E6F0FF !important;
    border: none !important;
}

div[data-testid="stForm"] {
    background-color: #E6F0FF !important;
    border-radius: 12px;
    padding: 12px;
}

div[data-testid="stTextInput"],
div[data-testid="stNumberInput"],
div[data-testid="stSelectbox"],
div[data-testid="stMultiSelect"] {
    background-color: #E6F0FF !important;
    border-radius: 12px;
    padding: 10px;
}

input, textarea {
    background-color: white !important;
    color: black !important;
    border: 1px solid #B0B0B0 !important; /* xám nhạt */
    border-radius: 8px;
}

button {
    background-color: #E6F0FF !important;
    border: 1px solid #B0B0B0 !important; /* xám nhạt */
    color: black !important;
    border-radius: 8px;
    box-shadow: none !important;
}

.stButton > button {
    border: 1px solid #B0B0B0 !important;
    background-color: #E6F0FF !important;
    color: black !important;
}

details, summary {
    background-color: #E6F0FF !important;
    border: none !important;
}

div[data-testid="stExpander"] {
    background-color: #E6F0FF !important;
    border-radius: 12px;
}

* {
    box-shadow: none !important;
div[data-baseweb="select"] {
    background-color: white !important;
}

div[data-baseweb="select"] > div {
    background-color: white !important;
    color: black !important;
    border: none !important;
    box-shadow: none !important;
}

div[data-baseweb="select"] div {
    border: none !important;
    box-shadow: none !important;
}

div[data-baseweb="popover"] {
    background-color: #d9d9d9 !important;
    border: none !important;
    box-shadow: none !important;
}

ul[role="listbox"] {
    background-color: #d9d9d9 !important;
    border: none !important;
    box-shadow: none !important;
}

li[role="option"] {
    background-color: #d9d9d9 !important;
    color: black !important;
    border: none !important;
}

li[role="option"]:hover {
    background-color: #cfcfcf !important;
}
header {
    background-color: #000000 !important;
}

div[data-testid="stToolbar"] {
    background-color: #000000 !important;
    border: 1px solid #333333 !important;
    border-radius: 0px !important;
    padding: 4px;
}

div[data-testid="stToolbar"] button {
    background-color: #d9d9d9 !important;  /* xám nổi bật */
    border: 1px solid #b0b0b0 !important;
    color: black !important;
    border-radius: 0px !important;
}

div[data-testid="stToolbar"] button:hover {
    background-color: #cfcfcf !important;
}

#MainMenu, footer {
    background-color: #000000 !important;
    border: none !important;
}
            
</style>
""", unsafe_allow_html=True)

def safe_str(x):
    if x is None:
        return "None"
    x = str(x).strip()
    return x if x else "None"

def safe_int(x):
    try:
        return int(re.search(r"\d+", str(x)).group())
    except:
        return 0

def safe_float(x):
    try:
        return float(str(x).replace(",", "."))
    except:
        return 0.0

def safe_gender(x):
    if not x:
        return "None"
    x = str(x).lower()
    if "male" in x and "female" not in x:
        return "Male"
    if "female" in x:
        return "Female"
    return "None"

init_state = {
    "patient_age": 0,
    "patient_gender": "None",
    "patient_income": 0.0,
    "patient_employment": "None",
    "patient_marital": "None",
    "provider_specialty": "None",
    "claim_type": "None",
    "claim_submission_method": "None",
    "diagnosis": "None",
    "procedure": "None",
    "claim_status": "None",
    "claim_amount": 0.0,
    "raw_text": ""
}

for k, v in init_state.items():
    if k not in st.session_state:
        st.session_state[k] = v

st.title("Health Insurance Claim Approval System")

left, right = st.columns([1, 1.2])

with left:
    st.subheader("Upload Document")

    uploaded_file = st.file_uploader("Select Image", type=["png", "jpg", "jpeg"])

    if uploaded_file:
        file_bytes = uploaded_file.getvalue()
        image = Image.open(io.BytesIO(file_bytes))
        st.image(image, use_container_width=True)

        if st.button("Run OCR"):
            mime_type = mimetypes.guess_type(uploaded_file.name)[0] or "image/jpeg"

            files = {
                "file": (uploaded_file.name, file_bytes, mime_type)
            }

            response = requests.post(OCR_API_URL, files=files, timeout=60)

            if response.status_code != 200:
                st.error(response.text)
                st.stop()

            data = response.json()
            st.session_state["raw_text"] = data.get("raw_text", "")
            extracted = data.get("extracted_data", {})

            st.session_state["patient_age"] = safe_int(extracted.get("PatientAge"))
            st.session_state["patient_gender"] = safe_gender(extracted.get("PatientGender"))
            st.session_state["patient_income"] = safe_float(extracted.get("PatientIncome"))
            st.session_state["patient_employment"] = safe_str(extracted.get("PatientEmploymentStatus"))
            st.session_state["patient_marital"] = safe_str(extracted.get("PatientMaritalStatus"))
            st.session_state["provider_specialty"] = safe_str(extracted.get("ProviderSpecialty"))
            st.session_state["claim_type"] = safe_str(extracted.get("ClaimType"))
            st.session_state["claim_submission_method"] = safe_str(extracted.get("ClaimSubmissionMethod"))
            st.session_state["diagnosis"] = safe_str(extracted.get("DiagnosisCode"))
            st.session_state["procedure"] = safe_str(extracted.get("ProcedureCode"))
            st.session_state["claim_status"] = safe_str(extracted.get("ClaimStatus"))
            st.session_state["claim_amount"] = safe_float(extracted.get("ClaimAmount"))

with right:
    st.subheader("Claim Information")

    with st.form("claim_form"):
        c1, c2 = st.columns(2)

        with c1:
            patient_age = st.number_input("Age", 0, 120, key="patient_age")
            patient_gender = st.selectbox("Gender", ["Male", "Female", "None"], key="patient_gender")
            patient_income = st.number_input("Income", key="patient_income")
            patient_employment = st.text_input("Employment", key="patient_employment")
            provider_specialty = st.text_input("Provider Specialty", key="provider_specialty")
            claim_type = st.text_input("Claim Type", key="claim_type")

        with c2:
            patient_marital = st.text_input("Marital Status", key="patient_marital")
            diagnosis = st.text_input("Diagnosis Code", key="diagnosis")
            procedure = st.text_input("Procedure Code", key="procedure")
            claim_submission_method = st.text_input("Submission Method", key="claim_submission_method")
            claim_status = st.text_input("Claim Status", key="claim_status")

        claim_amount = st.number_input("Amount", key="claim_amount")

        submitted = st.form_submit_button("Submit Claim")

if submitted:
    query = f"""
RAW:{st.session_state.get("raw_text","")}

Age:{patient_age}
Gender:{patient_gender}
Income:{patient_income}
Employment:{patient_employment}
Marital:{patient_marital}
Specialty:{provider_specialty}
Type:{claim_type}
Submit:{claim_submission_method}
Diagnosis:{diagnosis}
Procedure:{procedure}
Status:{claim_status}
Amount:{claim_amount}
"""

    res = requests.get(API_URL, params={"query": query})

    if res.status_code == 200:
        result = res.json()
        decision = result.get("decision", "Pending")

        if decision == "Approved":
            st.success(decision)
        elif decision == "Denied":
            st.error(decision)
        else:
            st.warning(decision)

        st.json(result)
    else:
        st.error("API error")

with st.expander("OCR Raw Text"):
    st.text(st.session_state.get("raw_text", ""))

with st.expander("Extracted Data"):
    st.json({
        "patient_age": st.session_state["patient_age"],
        "patient_gender": st.session_state["patient_gender"],
        "patient_income": st.session_state["patient_income"],
        "patient_employment": st.session_state["patient_employment"],
        "patient_marital": st.session_state["patient_marital"],
        "provider_specialty": st.session_state["provider_specialty"],
        "claim_type": st.session_state["claim_type"],
        "claim_submission_method": st.session_state["claim_submission_method"],
        "diagnosis": st.session_state["diagnosis"],
        "procedure": st.session_state["procedure"],
        "claim_status": st.session_state["claim_status"],
        "claim_amount": st.session_state["claim_amount"]
    })