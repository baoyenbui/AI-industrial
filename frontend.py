import streamlit as st
import requests
from PIL import Image
import io
import re
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
section { background-color: transparent !important; }
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
h1, h2, h3, h4, h5, h6, p, span, label { color: black !important; }
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
div[data-baseweb="select"] svg { fill: #666 !important; }
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
li[role="option"] { background-color: white !important; color: black !important; }
li[role="option"]:hover { background-color: #E6F0FF !important; }
div[data-baseweb="input"] {
    border: 1px solid #B0B0B0 !important;
    border-radius: 8px !important;
}
button[data-baseweb="button"] { border: none !important; box-shadow: none !important; }
div[data-testid="stJsonContainer"] {
    background: white !important;
    color: black !important;
    border: 1px solid #B0B0B0 !important;
    border-radius: 8px !important;
}
pre { background: white !important; color: black !important; }
div[data-testid="stFormSubmitButton"] {
    display: flex !important;
    justify-content: center !important;
    margin-top: 12px;
}
div[data-testid="stFormSubmitButton"] button {
    width: 180px !important;
    border: 2px solid black !important;
    background: white !important;
    color: black !important;
    border-radius: 10px !important;
    font-weight: 600;
}
div[data-testid="stFormSubmitButton"] button:hover { background-color: #f5f5f5 !important; }
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
div[data-testid="stNumberInput"] label { margin-bottom: 4px !important; }
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
input::placeholder, textarea::placeholder { color: rgba(0,0,0,0.35) !important; }

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
    position: relative;
    vertical-align: middle;
}
.info-icon .tooltip-box {
    display: none;
    position: absolute;
    left: 22px;
    top: -6px;
    z-index: 9999;
    background: white;
    color: #111111;
    border: 1px solid #CCCCCC;
    border-radius: 8px;
    padding: 10px 13px;
    width: 230px;
    font-size: 12px;
    font-weight: 400;
    line-height: 1.6;
    box-shadow: 0 4px 18px rgba(0,0,0,0.12);
    pointer-events: none;
    white-space: normal;
}
.info-icon:hover .tooltip-box,
.info-icon:focus .tooltip-box {
    display: block;
}
.info-icon:hover, .info-icon:focus, .info-icon:active {
    outline: none !important;
    box-shadow: none !important;
}
.field-error-hint {
    color: #CC0000;
    font-size: 11px;
    margin-top: 2px;
    margin-bottom: 6px;
}

div[data-testid="stNumberInput"] input[value="0"],
div[data-testid="stNumberInput"] input[value="0.0"],
div[data-testid="stNumberInput"] input[value="0.00"] {
    color: rgba(0,0,0,0.35) !important;
    -webkit-text-fill-color: rgba(0,0,0,0.35) !important;
}
div[data-testid="stNumberInput"] input:focus {
    color: black !important;
    -webkit-text-fill-color: black !important;
}


.img-strip {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 10px;
}
.img-thumb-wrapper {
    position: relative;
    display: inline-block;
}
.img-thumb-label {
    font-size: 10px;
    color: #555;
    text-align: center;
    margin-top: 2px;
    max-width: 80px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
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


def field(label, help_html, widget):
    error_fields = st.session_state.get("error_fields", []) if st.session_state.get("_show_errors", False) else []
    has_error = label in error_fields
    color = "#CC0000" if has_error else "inherit"
    icon_border = f"border-color:{color};color:{color}"
    error_star = "<span style='color:#CC0000'> *</span>" if has_error else ""

    st.markdown(
        f"<div style='margin-bottom:4px;font-weight:600;color:{color};line-height:1.6'>"
        f"{label}{error_star}"
        f"<span class='info-icon' style='{icon_border}' tabindex='0'>i"
        f"<span class='tooltip-box'>{help_html}</span>"
        f"</span>"
        f"</div>",
        unsafe_allow_html=True
    )
    result = widget(label=label, help=None, label_visibility="collapsed")
    return result


def normalize_gender(x):
    x = clean_text(x)
    if not x:
        return "Male"
    x = x.lower()
    if "male" in x and "female" not in x:
        return "Male"
    if "female" in x:
        return "Female"
    if "other" in x:
        return "Other"
    return "Male"


def merge_ocr_results(results: list[dict]) -> dict:
    NUMERIC_SUM_FIELDS = {"ClaimAmount", "PatientIncome"}
    NUMERIC_FIRST_FIELDS = {"PatientAge"}

    merged: dict = {}
    for data in results:
        for key, val in data.items():
            if key in NUMERIC_SUM_FIELDS:
                merged[key] = merged.get(key, 0.0) + safe_float(val)
            elif key in NUMERIC_FIRST_FIELDS:
                if not merged.get(key):
                    v = safe_int(val)
                    if v:
                        merged[key] = v
            else:
                existing = clean_text(merged.get(key, ""))
                incoming = clean_text(val)
                if not existing and incoming:
                    merged[key] = incoming
    return merged

init_state = {
    "patient_age": 0,
    "patient_gender": "Male",
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
    "raw_text": "",
    "error_fields": [],
    "_uploaded_images": [],
}

for k, v in init_state.items():
    st.session_state.setdefault(k, v)

if "_last_submitted" not in st.session_state:
    st.session_state["_last_submitted"] = False

st.title("Health Insurance Claim Approval System")

left, right = st.columns([1, 1.2])

with left:
    st.markdown("<h3 style='margin-bottom:10px;'>Upload Documents</h3>", unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Select Images (multiple allowed)",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,   
    )

    if uploaded_files:
        st.session_state["_uploaded_images"] = [
            {"bytes": f.getvalue(), "name": f.name}
            for f in uploaded_files
        ]

    images_data = st.session_state.get("_uploaded_images", [])

    if images_data:
        n = len(images_data)
        st.markdown(
            f"<div style='font-size:12px;color:#555;margin-bottom:6px;'>"
            f"{n} image{'s' if n > 1 else ''} selected</div>",
            unsafe_allow_html=True,
        )

        thumb_cols = st.columns(min(n, 4))  
        for idx, img_info in enumerate(images_data):
            col_idx = idx % 4
            with thumb_cols[col_idx]:
                img = Image.open(io.BytesIO(img_info["bytes"]))
                w = 80
                h = int(w * img.height / img.width)
                img_resized = img.resize((w, h), Image.LANCZOS)
                st.image(img_resized, width=w, caption=img_info["name"][:14])

        col_btn = st.columns([1, 1, 1])
        with col_btn[1]:
            run_ocr = st.button(
                f"Run OCR ({n} image{'s' if n > 1 else ''})",
                use_container_width=True,
            )

        if run_ocr:
            all_extracted: list[dict] = []
            ocr_errors: list[str] = []

            progress = st.progress(0, text="Running OCR…")
            for i, img_info in enumerate(images_data):
                progress.progress((i) / n, text=f"Processing {img_info['name']} ({i+1}/{n})…")
                try:
                    res = requests.post(
                        OCR_API_URL,
                        files={"file": (img_info["name"], img_info["bytes"])},
                        timeout=30,
                    )
                    if res.status_code == 200:
                        data = res.json().get("extracted_data", {})
                        all_extracted.append(data)
                        if DEBUG:
                            st.write(f"OCR [{img_info['name']}] raw:", res.json())
                    else:
                        ocr_errors.append(f"{img_info['name']}: HTTP {res.status_code}")
                except Exception as e:
                    ocr_errors.append(f"{img_info['name']}: {e}")

            progress.progress(1.0, text="Done!")

            if ocr_errors:
                for err in ocr_errors:
                    st.warning(f"OCR error — {err}")

            if all_extracted:
                merged = merge_ocr_results(all_extracted)

                if DEBUG:
                    st.write("Merged OCR data:", merged)

                st.session_state["patient_age"]               = safe_int(merged.get("PatientAge"))
                st.session_state["patient_gender"]            = normalize_gender(merged.get("PatientGender"))
                st.session_state["patient_income"]            = safe_float(merged.get("PatientIncome"))
                st.session_state["patient_employment"]        = clean_text(merged.get("PatientEmploymentStatus"))
                st.session_state["patient_marital"]           = clean_text(merged.get("PatientMaritalStatus"))
                st.session_state["provider_specialty"]        = clean_text(merged.get("ProviderSpecialty"))
                st.session_state["claim_type"]                = clean_text(merged.get("ClaimType"))
                st.session_state["claim_submission_method"]   = clean_text(merged.get("ClaimSubmissionMethod"))
                st.session_state["diagnosis"]                 = clean_text(merged.get("DiagnosisCode"))
                st.session_state["procedure"]                 = clean_text(merged.get("ProcedureCode"))
                st.session_state["claim_status"]              = clean_text(merged.get("ClaimStatus"))
                st.session_state["claim_amount"]              = safe_float(merged.get("ClaimAmount"))
                st.session_state["error_fields"]              = []
                st.session_state["_show_errors"]              = False
                st.rerun()

with right:
    st.markdown("<h3 style='margin-bottom:10px;'>Claim Information</h3>", unsafe_allow_html=True)

    with st.form("claim_form"):
        c1, c2 = st.columns(2)

        with c1:
            field(
                "Age",
                "<b>Patient age</b> (in years).<br>"
                "Enter a value between 0 and 120.",
                lambda label, help, label_visibility: st.number_input(
                    label, min_value=0, max_value=120,
                    step=1,
                    key="patient_age", help=help, label_visibility=label_visibility)
            )

            field(
                "Gender",
                "<b>Patient gender</b>.",
                lambda label, help, label_visibility: st.selectbox(
                    label, ["Male", "Female", "Other"],
                    key="patient_gender", help=help, label_visibility=label_visibility)
            )

            field(
                "Income (USD/month)",
                "<b>Monthly income</b> before tax.",
                lambda label, help, label_visibility: st.number_input(
                    label, min_value=0.0,
                    step=100.0, format="%.0f",
                    key="patient_income", help=help, label_visibility=label_visibility)
            )

            field(
                "Employment",
                "<b>Employment status</b> of the patient (e.g. employed, unemployed, student, retired).",
                lambda label, help, label_visibility: st.text_input(
                    label, placeholder="e.g. employed",
                    key="patient_employment", help=help, label_visibility=label_visibility)
            )

            field(
                "Provider Specialty",
                "<b>Doctor's medical specialty</b> (e.g. cardiology, orthopedics, pediatrics).",
                lambda label, help, label_visibility: st.text_input(
                    label, placeholder="e.g. cardiology",
                    key="provider_specialty", help=help, label_visibility=label_visibility)
            )

            field(
                "Claim Type",
                "<b>Type of insurance claim</b> (e.g. medical, dental, vision, pharmacy).",
                lambda label, help, label_visibility: st.text_input(
                    label, placeholder="e.g. medical",
                    key="claim_type", help=help, label_visibility=label_visibility)
            )

        with c2:
            field(
                "Marital Status",
                "<b>Marital status</b> of the patient (e.g. single, married, divorced, widowed).",
                lambda label, help, label_visibility: st.text_input(
                    label, placeholder="e.g. single",
                    key="patient_marital", help=help, label_visibility=label_visibility)
            )

            field(
                "Diagnosis Code",
                "<b>Medical diagnosis code (ICD-10)</b> provided by the doctor.",
                lambda label, help, label_visibility: st.text_input(
                    label, placeholder="e.g. J18.9",
                    key="diagnosis", help=help, label_visibility=label_visibility)
            )

            field(
                "Procedure Code",
                "<b>Treatment procedure code (CPT)</b> performed during care.",
                lambda label, help, label_visibility: st.text_input(
                    label, placeholder="e.g. 99213",
                    key="procedure", help=help, label_visibility=label_visibility)
            )

            field(
                "Submission Method",
                "<b>How the claim was submitted</b> (online, paper, hospital).",
                lambda label, help, label_visibility: st.text_input(
                    label, placeholder="e.g. online",
                    key="claim_submission_method", help=help, label_visibility=label_visibility)
            )

            field(
                "Claim Status",
                "<b>Current status</b> of the claim (pending, approved, denied, appealed).",
                lambda label, help, label_visibility: st.text_input(
                    label, placeholder="e.g. pending",
                    key="claim_status", help=help, label_visibility=label_visibility)
            )

            field(
                "Amount (USD)",
                "<b>Total claim amount</b> before insurance adjustments.<br>"
                "When multiple invoices are uploaded, amounts are summed automatically.",
                lambda label, help, label_visibility: st.number_input(
                    label, min_value=0.0,
                    step=100.0, format="%.0f",
                    key="claim_amount", help=help, label_visibility=label_visibility)
            )

        c_btn2 = st.columns([1, 2, 1])
        with c_btn2[1]:
            submitted = st.form_submit_button("Submit Claim", use_container_width=True)

if submitted:
    st.session_state["error_fields"] = []

    def is_empty(v):
        return v is None or (isinstance(v, str) and v.strip() in ["", "None", "Choose an option"])

    def is_invalid_number(v, allow_zero=False):
        try:
            if v is None:
                return True
            v = float(v)
            return v < 0 if allow_zero else v <= 0
        except:
            return True

    def is_invalid_text(v):
        if v is None:
            return True
        v = str(v).strip()
        return v in ["", "None"] or len(v) < 2

    def normalize_code(v):
        if v is None:
            return ""
        return re.sub(r"[^\w\.]", "", str(v)).strip()

    def safe_int(x):
        if x is None:
            return 0
        m = re.findall(r"\d+", str(x))
        return int(m[0]) if m else 0

    def safe_float(x):
        try:
            return float(str(x).replace(",", "."))
        except:
            return 0.0

    required_fields = {
        "Age": st.session_state.get("patient_age"),
        "Gender": st.session_state.get("patient_gender"),
        "Income (USD/month)": st.session_state.get("patient_income"),
        "Employment": st.session_state.get("patient_employment"),
        "Provider Specialty": st.session_state.get("provider_specialty"),
        "Claim Type": st.session_state.get("claim_type"),
        "Marital Status": st.session_state.get("patient_marital"),
        "Diagnosis Code": st.session_state.get("diagnosis"),
        "Procedure Code": st.session_state.get("procedure"),
        "Submission Method": st.session_state.get("claim_submission_method"),
        "Claim Status": st.session_state.get("claim_status"),
        "Amount (USD)": st.session_state.get("claim_amount"),
    }

    missing = []
    invalid = []

    for k, v in required_fields.items():
        if is_empty(v):
            missing.append(k)
        else:
            if k == "Age":
                if is_invalid_number(v, allow_zero=False):
                    invalid.append(k)
            elif k in ["Income (USD/month)", "Amount (USD)"]:
                if is_invalid_number(v):
                    invalid.append(k)
            else:
                if is_invalid_text(v):
                    invalid.append(k)

    if missing or invalid:
        st.session_state["error_fields"] = missing + invalid
        st.session_state["_show_errors"] = True

        if len(missing + invalid) == 1:
            st.error(f"Please check the field: **{(missing + invalid)[0]}**")
        elif len(missing + invalid) <= 3:
            st.error(f"Please check: **{', '.join(missing + invalid)}**")
        else:
            st.error(f"Please fill in all required fields before submitting. ({len(missing + invalid)} fields incomplete)")

        st.rerun()

    st.session_state["error_fields"] = []
    st.session_state["_show_errors"] = False

    age_raw = st.session_state.get("patient_age")

    payload = {
        "PatientAge": safe_int(age_raw),
        "PatientGender": st.session_state.get("patient_gender"),
        "PatientIncome": safe_float(st.session_state.get("patient_income")),
        "PatientEmploymentStatus": st.session_state.get("patient_employment"),
        "ProviderSpecialty": st.session_state.get("provider_specialty"),
        "ClaimType": st.session_state.get("claim_type"),
        "ClaimAmount": safe_float(st.session_state.get("claim_amount")),
        "DiagnosisCode": normalize_code(st.session_state.get("diagnosis")),
        "ProcedureCode": normalize_code(st.session_state.get("procedure")),
        "ClaimSubmissionMethod": st.session_state.get("claim_submission_method"),
        "ClaimStatus": st.session_state.get("claim_status"),
        "PatientMaritalStatus": st.session_state.get("patient_marital"),
    }

    REASON_TO_FIELD = {
        "Missing diagnosis": "Diagnosis Code",
        "Income is 0": "Income (USD/month)",
        "missing_required_fields": "one or more required fields",
        "validation_failed": "one or more fields",
    }

    try:
        res = requests.post(API_URL, json=payload, timeout=20)

        if not res.headers.get("content-type", "").startswith("application/json"):
            st.error("Invalid API response format")
            st.stop()

        result = res.json()

    except:
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

        explanation = REASON_TO_FIELD.get(reason, reason or "No explanation provided")
        st.write(f"Explanation: {explanation}")

    else:
        st.error("Unknown response from server")