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

/* SUBMIT BUTTON CENTER */
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

/* BUTTON GLOBAL */
div[data-testid="stButton"] button {
    width: 180px !important;
    background: white !important;
    color: black !important;
    border: 2px solid black !important;
    border-radius: 10px !important;
}

/* FILE UPLOADER BUTTON */
div[data-testid="stFileUploader"] button {
    background: white !important;
    color: black !important;
    border: 2px solid black !important;
    border-radius: 10px !important;
    box-shadow: none !important;
}

/* NUMBER INPUT FIX */
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

input::placeholder,
textarea::placeholder {
    color: rgba(0, 0, 0, 0.4) !important;
}
            
.info-icon {
    display: inline-flex;
    align-items: center;
    justify-content: center;

    width: 16px;
    height: 16px;
    margin-left: 6px;

    border-radius: 50%;
    border: 1px solid black;

    font-size: 11px;
    font-weight: 600;
    color: black;

    cursor: help;

    outline: none !important;
    box-shadow: none !important;
}

.info-icon:hover,
.info-icon:focus,
.info-icon:active {
    outline: none !important;
    box-shadow: none !important;
}

div[data-testid="stFormSubmitButton"] {
    display: flex !important;
    justify-content: center !important;
    margin-top: 12px;
}

div[data-testid="stFormSubmitButton"] button {
    width: 180px !important;
    background-color: white !important;
    color: black !important;
    border: 1.5px solid black !important;
    border-radius: 8px !important;
    font-weight: 600;
}

div[data-testid="stFormSubmitButton"] button:hover {
    background-color: #f5f5f5 !important;
}

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


def field(label, help_text, widget):
    st.markdown(
        f"<div style='margin-bottom:-8px;font-weight:600'>{label} "
        f"<span class='info-icon' title='{help_text}'>i</span></div>",
        unsafe_allow_html=True
    )
    return widget(label="", help=None)

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
            field(
                "Age",
                "Patient age in years (0–120). If unknown, leave default 0.",
                lambda label, help: st.number_input(label, min_value=0, max_value=120, key="patient_age", help=help)
            )

            field(
                "Gender",
                "Select Male / Female / Other. Required for risk scoring.",
                lambda label, help: st.selectbox(label, ["Male", "Female", "Other"], key="patient_gender", help=help)
            )

            field(
                "Income",
                "Monthly income in USD. If unknown, keep 0.",
                lambda label, help: st.number_input(label, min_value=0.0, key="patient_income", help=help)
            )

            field(
                "Employment",
                "Job status like employed, unemployed, student, retired.",
                lambda label, help: st.text_input(label, key="patient_employment", help=help)
            )

            field(
                "Provider Specialty",
                "Medical department of doctor (e.g. cardiology).",
                lambda label, help: st.text_input(label, key="provider_specialty", help=help)
            )

            field(
                "Claim Type",
                "Type of insurance claim (medical, dental, etc.).",
                lambda label, help: st.text_input(label, key="claim_type", help=help)
            )

        with c2:
            field(
                "Marital Status",
                "single / married / divorced / widowed.",
                lambda label, help: st.text_input(label, key="patient_marital", help=help)
            )

            field(
                "Diagnosis Code",
                "ICD or internal medical diagnosis code.",
                lambda label, help: st.text_input(label, key="diagnosis", help=help)
            )

            field(
                "Procedure Code",
                "Medical procedure code performed in treatment.",
                lambda label, help: st.text_input(label, key="procedure", help=help)
            )

            field(
                "Submission Method",
                "How claim was submitted: online, paper, hospital.",
                lambda label, help: st.text_input(label, key="claim_submission_method", help=help)
            )

            field(
                "Claim Status",
                "Current processing status of claim.",
                lambda label, help: st.text_input(label, key="claim_status", help=help)
            )

            field(
                "Amount",
                "Total claimed amount in USD.",
                lambda label, help: st.number_input(label, min_value=0.0, key="claim_amount", help=help)
            )
            
        c_btn2 = st.columns([1, 2, 1])
        with c_btn2[1]:
            submitted = st.form_submit_button("Submit Claim", use_container_width=True)

if submitted:

    def is_empty(v):
        return v is None or str(v).strip() in ["", "None", "Choose an option"]

    def is_invalid_number(v):
        try:
            return v is None or float(v) <= 0
        except:
            return True

    def is_invalid_text(v):
        return v is not None and str(v).strip() not in ["", "None"] and len(str(v).strip()) < 2

    required_fields = {
        "Age": st.session_state.get("patient_age"),
        "Gender": st.session_state.get("patient_gender"),
        "Income": st.session_state.get("patient_income"),
        "Employment": st.session_state.get("patient_employment"),
        "Provider Specialty": st.session_state.get("provider_specialty"),
        "Claim Type": st.session_state.get("claim_type"),
        "Marital Status": st.session_state.get("patient_marital"),
        "Diagnosis Code": st.session_state.get("diagnosis"),
        "Procedure Code": st.session_state.get("procedure"),
        "Submission Method": st.session_state.get("claim_submission_method"),
        "Claim Status": st.session_state.get("claim_status"),
        "Amount": st.session_state.get("claim_amount"),
    }

    missing = []
    invalid = []

    for k, v in required_fields.items():
        if is_empty(v):
            missing.append(k)
        else:
            if k in ["Age", "Income", "Amount"]:
                if is_invalid_number(v):
                    invalid.append(k)
            else:
                if is_invalid_text(v):
                    invalid.append(k)

    if missing or invalid:
        st.error("Input validation failed")

        if missing:
            st.write("Missing fields:")
            for f in missing:
                st.write(f"- {f}")

        if invalid:
            st.write("Invalid format fields:")
            for f in invalid:
                st.write(f"- {f}")

        st.stop()

    payload = {
        "PatientAge": st.session_state.get("patient_age"),
        "PatientGender": st.session_state.get("patient_gender"),
        "PatientIncome": st.session_state.get("patient_income"),
        "PatientEmploymentStatus": st.session_state.get("patient_employment"),
        "ProviderSpecialty": st.session_state.get("provider_specialty"),
        "ClaimType": st.session_state.get("claim_type"),
        "ClaimAmount": st.session_state.get("claim_amount"),
    }

    try:
        res = requests.post(API_URL, json=payload, timeout=20)

        if not res.headers.get("content-type", "").startswith("application/json"):
            st.error("Invalid API response format")
            st.stop()

        result = res.json()

    except Exception:
        st.error("System error: cannot connect to API")
        st.stop()

    status = result.get("status")
    decision = result.get("decision", "Pending")
    reason = result.get("reason", "")
    confidence = result.get("confidence")

    if status == "invalid_input":
        st.error("Validation error")
        for e in result.get("errors", []):
            st.write(f"- {e['field']}: {e['message']}")

    elif status == "system_error":
        st.error("System error occurred")

    elif status == "failed":
        st.warning("OCR failed or no usable text extracted")

    elif status == "ok":
        st.subheader("Decision Result")

        if decision == "Approved":
            st.success(f"Status: {decision}")
        elif decision == "Denied":
            st.error(f"Status: {decision}")
        else:
            st.warning(f"Status: {decision}")

        st.write(f"Confidence: {confidence if confidence is not None else 'N/A'}")
        st.write(f"Explanation: {reason if reason else 'No explanation provided'}")

    else:
        st.error("Unknown response from server")