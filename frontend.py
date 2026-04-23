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

div.block-container {
    background-color: white !important;
    padding-top: 2rem !important;
    padding-bottom: 2rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    max-width: 1200px;
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

input:not([type="hidden"]), textarea {
    background-color: white !important;
    color: black !important;
    border: 1px solid #B0B0B0 !important;
    border-radius: 8px !important;
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

div[data-baseweb="select"] input {
    opacity: 0 !important;
    position: absolute !important;
    pointer-events: none !important;
    height: 0 !important;
    width: 0 !important;
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
    color: rgba(0, 0, 0, 0.4) !important;
}

textarea::placeholder {
    color: rgba(0, 0, 0, 0.4) !important;
}
</style>
""", unsafe_allow_html=True)


def clean_text(x):
    if x is None:
        return ""
    x = str(x)
    x = unicodedata.normalize("NFKC", x)
    x = re.sub(r"[\u200b-\u200f\ufeff\u00a0]", "", x)
    x = re.sub(r"[\x00-\x1f\x7f]", "", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x

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
    "patient_gender": None,
    "patient_income": 0.0,
    "patient_employment": None,
    "patient_marital": None,
    "provider_specialty": None,
    "claim_type": None,
    "claim_submission_method": None,
    "diagnosis": None,
    "procedure": None,
    "claim_status": None,
    "claim_amount": 0.0,
    "raw_text": ""
}

for k, v in init_state.items():
    st.session_state.setdefault(k, v)

st.title("Health Insurance Claim Approval System")

left, right = st.columns([1, 1.2])

with left:
    st.markdown("<h3 style='margin-bottom:10px;'>Upload Document</h3>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Select Image", type=["png", "jpg", "jpeg"])

    if uploaded_file:
        image = Image.open(io.BytesIO(uploaded_file.getvalue()))

        w = 285
        h = int(w * image.height / image.width)
        image = image.resize((w, h), Image.LANCZOS)

        col_img = st.columns([1, 3, 1])

        with col_img[1]:
            st.image(image, width=w, caption="Uploaded Document")
            
        col_btn = st.columns([1, 1, 1])
        with col_btn[1]:
            run_ocr = st.button("Run OCR", use_container_width=True)

        if run_ocr:
            res = requests.post(
                OCR_API_URL,
                files={"file": uploaded_file}
            )

            if res.status_code == 200:
                data = res.json().get("extracted_data", {})

                st.session_state["patient_age"] = safe_int(data.get("PatientAge"))
                st.session_state["patient_gender"] = normalize_gender(data.get("PatientGender"))
                st.session_state["patient_income"] = safe_float(data.get("PatientIncome"))
                st.session_state["patient_employment"] = clean_text(data.get("PatientEmploymentStatus"))
                st.session_state["patient_marital"] = clean_text(data.get("PatientMaritalStatus"))
                st.session_state["provider_specialty"] = clean_text(data.get("ProviderSpecialty"))
                st.session_state["claim_type"] = clean_text(data.get("ClaimType"))
                st.session_state["claim_submission_method"] = clean_text(data.get("ClaimSubmissionMethod"))
                st.session_state["diagnosis"] = clean_text(data.get("DiagnosisCode"))
                st.session_state["procedure"] = clean_text(data.get("ProcedureCode"))
                st.session_state["claim_status"] = clean_text(data.get("ClaimStatus"))
                st.session_state["claim_amount"] = safe_float(data.get("ClaimAmount"))

with right:
    st.markdown("<h3 style='margin-bottom:10px;'>Claim Information</h3>", unsafe_allow_html=True)

    with st.form("claim_form"):
        c1, c2 = st.columns(2)

        with c1:
            st.number_input("Age", 0, 120, key="patient_age")

            gender_options = ["Male", "Female", "Other"]

            st.selectbox(
                "Gender",
                options=gender_options,
                index=None,
                placeholder="Choose gender",
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
        
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        c_btn2 = st.columns([1, 2, 1])

        with c_btn2[1]:
            submitted = st.form_submit_button("Submit Claim", use_container_width=True)

if submitted:
    gender = st.session_state["patient_gender"]
    gender = gender if gender in ["Male", "Female", "Other"] else "None"

    query = f"""

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
            background: #ffffff;
            border: 1px solid rgba(0,0,0,0.08);
            border-radius: 16px;
            padding: 20px;
            margin-top: 18px;
            box-shadow: 0 6px 18px rgba(0,0,0,0.06);
        ">
            <h3 style="margin-bottom: 10px;">Decision Result</h3>
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