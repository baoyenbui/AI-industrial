import streamlit as st
import requests
from PIL import Image
import io
import re
import mimetypes
import unicodedata

API_URL = "http://127.0.0.1:8000/predict"
OCR_API_URL = "http://127.0.0.1:8000/ocr-claim"

st.set_page_config(page_title="Health Claim AI", layout="wide")

DEBUG = st.sidebar.toggle("Debug mode", value=False)

st.markdown("""
<style>
.stApp {
    background-color: white !important;
    border: none !important;
    box-shadow: none !important;
}

.block-container {
    background-color: white !important;
}

section {
    background-color: transparent !important;
}

div[data-testid="stFileUploader"],
div[data-testid="stForm"],
div[data-testid="stExpander"] {
    background-color: #E6F0FF !important;
    border-radius: 12px;
    padding: 12px;
}

input, textarea {
    background-color: white !important;
    color: black !important;
    border: 1px solid #B0B0B0 !important;
    border-radius: 8px;
}

h1, h2, h3, h4, h5, h6, p, span, label {
    color: black !important;
}

button, .stButton > button {
    background-color: #E6F0FF !important;
    border: 1px solid #B0B0B0 !important;
    color: black !important;
    border-radius: 8px;
}

div[data-baseweb="select"] > div {
    background: white !important;
    color: black !important;
    border: 1px solid #B0B0B0 !important;
    border-radius: 8px !important;
    box-shadow: none !important;
}

div[data-baseweb="select"] svg {
    fill: #666 !important;
}

ul[role="listbox"] {
    background-color: white !important;
    border: 1px solid #B0B0B0 !important;
}

li[role="option"] {
    background-color: white !important;
    color: black !important;
}

li[role="option"]:hover {
    background-color: #E6F0FF !important;
}

div[data-baseweb="input"] {
    border: 1px solid #B0B0B0 !important;
    border-radius: 8px !important;
}

button[data-baseweb="button"] {
    border: none !important;
    box-shadow: none !important;
}

div[data-testid="stJsonContainer"] {
    background: white !important;
    color: black !important;
    border: 1px solid #B0B0B0 !important;
    border-radius: 8px !important;
}

pre {
    background: white !important;
    color: black !important;
}

div[data-testid="stFormSubmitButton"] {
    display: flex !important;
    justify-content: center !important;
}

div[data-testid="stFormSubmitButton"] button {
    width: 180px !important;
    border: 2px solid black !important;
    background: white !important;
    color: black !important;
    border-radius: 10px !important;
}
            
div[data-testid="stButton"] button {
    width: 180px !important;
    background: white !important;
    color: black !important;
    border: 2px solid black !important;
    border-radius: 10px !important;
}

div[data-testid="stFileUploader"] button {
    background: white !important;
    color: black !important;
    border: 2px solid black !important;
    border-radius: 10px !important;
    box-shadow: none !important;
}

div[data-testid="stNumberInput"] {
    border: none !important;
    background: transparent !important;
    padding: 0 !important;
}

div[data-testid="stNumberInput"] label {
    margin-bottom: 4px !important;
}

div[data-testid="stNumberInput"] > div > div {
    background: white !important;
    border: 1px solid #B0B0B0 !important;
    border-radius: 8px !important;
    overflow: hidden !important;
}

div[data-testid="stNumberInput"] input {
    background: white !important;
    border: none !important;
    color: black !important;
    -webkit-text-fill-color: black !important;
}

div[data-testid="stNumberInput"] button {
    background: white !important;
    border-left: 1px solid #B0B0B0 !important;
}

input::placeholder {
    color: black !important;
    opacity: 1 !important;
}

textarea::placeholder {
    color: black !important;
    opacity: 1 !important;
}


</style>
""", unsafe_allow_html=True)

def clean_text(x):
    if x is None:
        return None
    x = str(x)
    x = unicodedata.normalize("NFKC", x)
    x = re.sub(r"[\u200b-\u200f\ufeff\u00a0]", "", x)
    x = re.sub(r"[\x00-\x1f\x7f]", "", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x if x else None


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


def normalize_gender(x):
    x = clean_text(x)
    if not x:
        return "Choose gender"
    x = x.lower()
    if "male" in x and "female" not in x:
        return "Male"
    if "female" in x:
        return "Female"
    if "other" in x:
        return "Other"
    return "Choose gender"


init_state = {
    "patient_age": 0,
    "patient_gender": "Choose gender",
    "patient_income": 0.0,
    "patient_employment": "",
    "patient_marital": "",
    "provider_specialty": "",
    "claim_type": "",
    "claim_submission_method": "",
    "diagnosis": "",
    "procedure": "",
    "claim_status": "",
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

        c_img = st.columns([1, 3, 1])
        with c_img[1]:
            st.image(image, use_container_width=True)

        c_btn = st.columns([1, 2, 1])
        with c_btn[1]:
            run_ocr = st.button("Run OCR", key="run_ocr_btn")

        if run_ocr:
            mime_type = mimetypes.guess_type(uploaded_file.name)[0] or "image/jpeg"
            files = {"file": (uploaded_file.name, file_bytes, mime_type)}
            response = requests.post(OCR_API_URL, files=files, timeout=60)

            if response.status_code != 200:
                st.error(response.text)
                st.stop()

            data = response.json()
            extracted = data.get("extracted_data", {})

            st.session_state["patient_age"] = safe_int(extracted.get("PatientAge"))
            st.session_state["patient_gender"] = normalize_gender(extracted.get("PatientGender"))
            st.session_state["patient_income"] = safe_float(extracted.get("PatientIncome"))
            st.session_state["patient_employment"] = clean_text(extracted.get("PatientEmploymentStatus")) or ""
            st.session_state["patient_marital"] = clean_text(extracted.get("PatientMaritalStatus")) or ""
            st.session_state["provider_specialty"] = clean_text(extracted.get("ProviderSpecialty")) or ""
            st.session_state["claim_type"] = clean_text(extracted.get("ClaimType")) or ""
            st.session_state["claim_submission_method"] = clean_text(extracted.get("ClaimSubmissionMethod")) or ""
            st.session_state["diagnosis"] = clean_text(extracted.get("DiagnosisCode")) or ""
            st.session_state["procedure"] = clean_text(extracted.get("ProcedureCode")) or ""
            st.session_state["claim_status"] = clean_text(extracted.get("ClaimStatus")) or ""
            st.session_state["claim_amount"] = safe_float(extracted.get("ClaimAmount"))

with right:
    st.subheader("Claim Information")

    with st.form("claim_form"):
        c1, c2 = st.columns(2)

        with c1:
            st.number_input("Age", 0, 120, key="patient_age")

            st.selectbox(
                "Gender",
                ["Choose gender", "Male", "Female", "Other"],
                key="patient_gender"
            )

            st.number_input("Income", key="patient_income")
            st.text_input("Employment", key="patient_employment", placeholder="None")
            st.text_input("Provider Specialty", key="provider_specialty", placeholder="None")
            st.text_input("Claim Type", key="claim_type", placeholder="None")

        with c2:
            st.text_input("Marital Status", key="patient_marital", placeholder="None")
            st.text_input("Diagnosis Code", key="diagnosis", placeholder="None")
            st.text_input("Procedure Code", key="procedure", placeholder="None")
            st.text_input("Submission Method", key="claim_submission_method", placeholder="None")
            st.text_input("Claim Status", key="claim_status", placeholder="None")
            st.number_input("Amount", key="claim_amount")

        c_btn2 = st.columns([1, 1, 1])
        with c_btn2[1]:
            submitted = st.form_submit_button("Submit Claim")

if submitted:
    gender = st.session_state["patient_gender"]
    if gender == "Choose gender":
        gender = "None"

    query = f"""
RAW:{st.session_state["raw_text"]}
Age:{st.session_state["patient_age"]}
Gender:{gender}
Income:{st.session_state["patient_income"]}
Employment:{st.session_state["patient_employment"] or "None"}
Marital:{st.session_state["patient_marital"] or "None"}
Specialty:{st.session_state["provider_specialty"] or "None"}
Type:{st.session_state["claim_type"] or "None"}
Submit:{st.session_state["claim_submission_method"] or "None"}
Diagnosis:{st.session_state["diagnosis"] or "None"}
Procedure:{st.session_state["procedure"] or "None"}
Status:{st.session_state["claim_status"] or "None"}
Amount:{st.session_state["claim_amount"]}
"""

    res = requests.get(API_URL, params={"query": query})

    if res.status_code == 200:
        result = res.json()

        decision = result.get("decision", "Pending")
        reason = result.get("reason", "")
        confidence = result.get("confidence", 0)

        if (not reason) or reason == "parse_error":
            reason = "Not enough information"

        st.markdown(f"""
        <div style="
            background:white;
            border:2px solid black;
            border-radius:12px;
            padding:16px;
            color:black;
            margin-top:12px;
        ">
            <h3>Decision Result</h3>
            <p><b>Status:</b> {decision}</p>
            <p><b>Confidence:</b> {confidence}</p>
            <p><b>Explanation:</b> {reason}</p>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.error("API error")

if DEBUG:
    with st.expander("Raw Text"):
        st.text(st.session_state["raw_text"])

    with st.expander("State"):
        st.json(st.session_state)