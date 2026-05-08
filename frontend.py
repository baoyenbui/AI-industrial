import streamlit as st
import requests
from PIL import Image
import io
import re
import unicodedata
from datetime import datetime

API_URL     = "http://127.0.0.1:8000/predict"
OCR_API_URL = "http://127.0.0.1:8000/ocr-claim"

st.set_page_config(page_title="Health Claim AI", layout="wide")

st.markdown("""
<style>
.stApp { background-color: white !important; border: none !important; box-shadow: none !important; }
div.block-container {
    background-color: white !important;
    padding-top: 2rem !important; padding-bottom: 2rem !important;
    padding-left: 2rem !important; padding-right: 2rem !important;
    max-width: 1200px;
}
section { background-color: transparent !important; }
div[data-testid="stFileUploader"],
div[data-testid="stForm"],
div[data-testid="stExpander"] {
    background-color: #E6F0FF !important; border-radius: 12px; padding: 12px;
}
input:not([type="hidden"]), textarea {
    background-color: white !important; color: black !important;
    border: 1px solid #B0B0B0 !important; border-radius: 8px !important;
}
h1, h2, h3, h4, h5, h6, p, span, label { color: black !important; }
button, .stButton > button {
    background-color: #E6F0FF !important; border: 1px solid #B0B0B0 !important;
    color: black !important; border-radius: 8px;
}
div[data-baseweb="select"] > div {
    background: white !important; color: black !important;
    border: 1px solid #B0B0B0 !important; border-radius: 8px !important; box-shadow: none !important;
}
div[data-baseweb="select"] svg { fill: #666 !important; }
div[data-baseweb="select"] input {
    opacity: 0 !important; position: absolute !important;
    pointer-events: none !important; height: 0 !important; width: 0 !important;
}
ul[role="listbox"] { background-color: white !important; border: 1px solid #B0B0B0 !important; }
li[role="option"] { background-color: white !important; color: black !important; }
li[role="option"]:hover { background-color: #E6F0FF !important; }
div[data-baseweb="input"] { border: 1px solid #B0B0B0 !important; border-radius: 8px !important; }
button[data-baseweb="button"] { border: none !important; box-shadow: none !important; }
div[data-testid="stJsonContainer"] {
    background: white !important; color: black !important;
    border: 1px solid #B0B0B0 !important; border-radius: 8px !important;
}
pre { background: white !important; color: black !important; }
div[data-testid="stFormSubmitButton"] {
    display: flex !important; justify-content: center !important; margin-top: 12px;
}
div[data-testid="stFormSubmitButton"] button {
    width: 180px !important; border: 2px solid black !important;
    background: white !important; color: black !important;
    border-radius: 10px !important; font-weight: 600;
}
div[data-testid="stFormSubmitButton"] button:hover { background-color: #f5f5f5 !important; }
div[data-testid="stButton"] button {
    width: 180px !important; background: white !important; color: black !important;
    border: 2px solid black !important; border-radius: 10px !important;
}
div[data-testid="stFileUploader"] button {
    background: white !important; color: black !important;
    border: 2px solid black !important; border-radius: 10px !important; box-shadow: none !important;
}
div[data-testid="stNumberInput"] { border: none !important; background: transparent !important; padding: 0 !important; }
div[data-testid="stNumberInput"] label { margin-bottom: 4px !important; }
div[data-testid="stNumberInput"] > div > div {
    background: white !important; border: 1px solid #B0B0B0 !important;
    border-radius: 8px !important; overflow: hidden !important;
}
div[data-testid="stNumberInput"] input {
    background: white !important; border: none !important;
    color: black !important; -webkit-text-fill-color: black !important;
}
div[data-testid="stNumberInput"] button { background: white !important; border-left: 1px solid #B0B0B0 !important; }
input::placeholder, textarea::placeholder { color: rgba(0,0,0,0.35) !important; }
.info-icon {
    display: inline-flex; align-items: center; justify-content: center;
    width: 16px; height: 16px; margin-left: 6px; border-radius: 50%;
    border: 1px solid black; font-size: 11px; font-weight: 600;
    color: black; cursor: help; outline: none !important; box-shadow: none !important;
    position: relative; vertical-align: middle;
}
.info-icon .tooltip-box {
    display: none; position: absolute; left: 22px; top: -6px; z-index: 9999;
    background: white; color: #111111; border: 1px solid #CCCCCC;
    border-radius: 8px; padding: 10px 13px; width: 230px;
    font-size: 12px; font-weight: 400; line-height: 1.6;
    box-shadow: 0 4px 18px rgba(0,0,0,0.12); pointer-events: none; white-space: normal;
}
.info-icon:hover .tooltip-box, .info-icon:focus .tooltip-box { display: block; }
.info-icon:hover, .info-icon:focus, .info-icon:active { outline: none !important; box-shadow: none !important; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def clean_text(x):
    if x is None:
        return ""
    x = str(x)
    x = unicodedata.normalize("NFKC", x)
    x = re.sub(r"[\u200b-\u200f\ufeff\u00a0]", "", x)
    x = re.sub(r"[\x00-\x1f\x7f]", "", x)
    return re.sub(r"\s+", " ", x).strip()


def safe_int(x):
    try:
        return int(re.search(r"\d+", str(x)).group())
    except Exception:
        return 0


def safe_float(x):
    try:
        return float(str(x).replace(",", "."))
    except Exception:
        return 0.0


def normalize_gender(x):
    x = clean_text(x)
    if not x or x == "Choose the gender":
        return ""
    xl = x.lower()
    if "female" in xl:
        return "Female"
    if "male" in xl:
        return "Male"
    return "Other"


def _is_blank(v, min_len=2):
    if v is None:
        return True
    return str(v).strip() in ("", "None") or len(str(v).strip()) < min_len


def merge_ocr_results(results: list[dict]) -> dict:
    NUMERIC_SUM   = {"ClaimAmount", "PatientIncome"}
    NUMERIC_FIRST = {"PatientAge"}
    merged: dict  = {}
    for data in results:
        for key, val in data.items():
            if key in NUMERIC_SUM:
                merged[key] = merged.get(key, 0.0) + safe_float(val)
            elif key in NUMERIC_FIRST:
                if not merged.get(key):
                    v = safe_int(val)
                    if v:
                        merged[key] = v
            else:
                if not clean_text(merged.get(key, "")) and clean_text(val):
                    merged[key] = clean_text(val)
    return merged


# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------

DEFAULTS = {
    "patient_age":            0,
    "patient_gender":         "Choose the gender",
    "patient_income":         0.0,
    "patient_employment":     "",
    "patient_marital":        "",
    "provider_specialty":     "",
    "claim_type":             "",
    "claim_submission_method":"",
    "diagnosis":              "",
    "procedure":              "",
    "claim_status":           "",
    "claim_amount":           0.0,
    "policy_number":          "",
    "date_of_service":        datetime.today().date(),
    "hospital_name":          "",
    "pre_auth_status":        "Choose an option",
    "error_fields":           [],   # list of field labels with errors
    "_submitted_once":        False, # đã bấm submit ít nhất 1 lần chưa
    "_api_result":            None,  # kết quả từ API sau submit thành công
    "_uploaded_images":       [],
}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


# ---------------------------------------------------------------------------
# Field renderer — highlight đỏ nếu đã submit và field có lỗi
# ---------------------------------------------------------------------------

def field(label, help_html, widget_fn):
    error_fields   = st.session_state.get("error_fields", [])
    submitted_once = st.session_state.get("_submitted_once", False)

    has_error  = submitted_once and (label in error_fields)
    color      = "#CC0000" if has_error else "inherit"
    icon_style = f"border-color:{color};color:{color}"
    star       = "<span style='color:#CC0000'>*</span>" if has_error else ""

    st.markdown(
        f"<div style='margin-bottom:4px;font-weight:600;color:{color};line-height:1.6'>"
        f"{label}{star}"
        f"<span class='info-icon' style='{icon_style}' tabindex='0'>i"
        f"<span class='tooltip-box'>{help_html}</span>"
        f"</span></div>",
        unsafe_allow_html=True,
    )
    return widget_fn(label=label, help=None, label_visibility="collapsed")


# ---------------------------------------------------------------------------
# Validate — trả về list label bị lỗi
# ---------------------------------------------------------------------------

def validate() -> list[str]:
    s      = st.session_state
    errors = []
    if s.patient_age <= 0:
        errors.append("Age")
    if s.patient_gender in ("Choose the gender", "", None):
        errors.append("Gender")
    if s.patient_income < 0:
        errors.append("Income (USD/month)")
    if _is_blank(s.patient_employment):
        errors.append("Employment")
    if s.claim_amount <= 0:
        errors.append("Total Claim Amount (USD)")
    if _is_blank(s.diagnosis):
        errors.append("Diagnosis Code")
    if _is_blank(s.hospital_name, min_len=3):
        errors.append("Hospital / Clinic Name")
    if s.pre_auth_status in ("Choose an option", "", None):
        errors.append("Pre-Authorization Status")
    return errors


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

st.title("Health Insurance Claim Approval System")

left, right = st.columns([1, 1.2])

# ── LEFT: upload + OCR ──────────────────────────────────────────────────────
with left:
    st.markdown("<h3 style='margin-bottom:10px;'>Upload Documents</h3>", unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Select Images (multiple allowed)",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        st.session_state["_uploaded_images"] = [
            {"bytes": f.getvalue(), "name": f.name} for f in uploaded_files
        ]
    elif not uploaded_files:
        st.session_state["_uploaded_images"] = []

    images_data = st.session_state.get("_uploaded_images", [])

    if images_data:
        n = len(images_data)
        st.markdown(
            f"<div style='text-align:center;font-size:13px;color:#555;margin:12px 0 16px 0;'>"
            f"{n} image{'s' if n > 1 else ''} selected</div>",
            unsafe_allow_html=True,
        )

        thumb_width  = 160 if n <= 2 else (120 if n <= 4 else 95)
        cols_per_row = min(n, 5)
        thumb_cols   = st.columns(cols_per_row)
        for idx, img_info in enumerate(images_data):
            with thumb_cols[idx % cols_per_row]:
                try:
                    img        = Image.open(io.BytesIO(img_info["bytes"]))
                    new_height = int(thumb_width * img.height / img.width)
                    st.image(img.resize((thumb_width, new_height), Image.LANCZOS),
                             width=thumb_width, caption=img_info["name"][:14])
                except Exception:
                    st.error(f"Cannot load {img_info['name']}")

        st.markdown("<div style='margin-bottom:28px;'></div>", unsafe_allow_html=True)

        col_btn = st.columns([1, 2, 1])
        with col_btn[1]:
            run_ocr = st.button(f"Run OCR ({n} image{'s' if n > 1 else ''})", use_container_width=True)

        if run_ocr:
            all_extracted: list[dict] = []
            ocr_errors:   list[str]   = []
            progress = st.progress(0, text="Running OCR…")

            for i, img_info in enumerate(images_data):
                progress.progress(i / n, text=f"Processing {img_info['name']} ({i+1}/{n})…")
                try:
                    res = requests.post(
                        OCR_API_URL,
                        files={"file": (img_info["name"], img_info["bytes"])},
                        timeout=30,
                    )
                    if res.status_code == 200:
                        all_extracted.append(res.json().get("extracted_data", {}))
                    else:
                        ocr_errors.append(f"{img_info['name']}: HTTP {res.status_code}")
                except Exception as e:
                    ocr_errors.append(f"{img_info['name']}: {e}")

            progress.progress(1.0, text="Done!")
            for err in ocr_errors:
                st.warning(f"OCR error — {err}")

            if all_extracted:
                m = merge_ocr_results(all_extracted)
                st.session_state.update({
                    "patient_age":             safe_int(m.get("PatientAge")),
                    "patient_gender":          normalize_gender(m.get("PatientGender")) or "Choose the gender",
                    "patient_income":          safe_float(m.get("PatientIncome")),
                    "patient_employment":      clean_text(m.get("PatientEmploymentStatus")),
                    "patient_marital":         clean_text(m.get("PatientMaritalStatus")),
                    "provider_specialty":      clean_text(m.get("ProviderSpecialty")),
                    "claim_type":              clean_text(m.get("ClaimType")),
                    "claim_submission_method": clean_text(m.get("ClaimSubmissionMethod")),
                    "diagnosis":               clean_text(m.get("DiagnosisCode")),
                    "procedure":               clean_text(m.get("ProcedureCode")),
                    "claim_status":            clean_text(m.get("ClaimStatus")),
                    "claim_amount":            safe_float(m.get("ClaimAmount")),
                    "policy_number":           clean_text(m.get("PolicyNumber")),
                    "hospital_name":           clean_text(m.get("HospitalName")),
                })
                pre_raw = clean_text(m.get("PreAuthorizationStatus"))
                if pre_raw in ("Yes", "No"):
                    st.session_state["pre_auth_status"] = pre_raw

                date_str = m.get("DateOfService")
                if date_str:
                    for fmt in ("%d %B %Y", "%B %d %Y", "%d/%m/%Y", "%m/%d/%Y",
                                "%Y-%m-%d", "%d-%m-%Y"):
                        try:
                            st.session_state["date_of_service"] = datetime.strptime(
                                str(date_str).strip(), fmt).date()
                            break
                        except Exception:
                            continue

                # OCR fill xong → reset lỗi cũ (user muốn xem lại form sạch)
                st.session_state["error_fields"]    = []
                st.session_state["_submitted_once"] = False
                st.session_state["_api_result"]     = None
                st.success("OCR completed — please review and submit.")
                st.rerun()
    else:
        st.markdown(
            "<div style='text-align:center;color:#888;font-size:14.5px;"
            "margin-top:50px;margin-bottom:50px;'><em>No images uploaded yet.</em></div>",
            unsafe_allow_html=True,
        )

# ── RIGHT: form ─────────────────────────────────────────────────────────────
with right:
    st.markdown("<h3 style='margin-bottom:10px;'>Claim Information</h3>", unsafe_allow_html=True)

    with st.form("claim_form"):
        c1, c2 = st.columns(2)

        with c1:
            field("Policy Number / Member ID", "Your policy number or member ID",
                  lambda label, help, label_visibility: st.text_input(
                      label, placeholder="e.g. POL123456789",
                      key="policy_number", help=help, label_visibility=label_visibility))

            field("Hospital / Clinic Name", "Name of hospital or clinic",
                  lambda label, help, label_visibility: st.text_input(
                      label, placeholder="e.g. Vinmec Central Park Hospital",
                      key="hospital_name", help=help, label_visibility=label_visibility))

            field("Age", "<b>Patient age</b> (0–120).",
                  lambda label, help, label_visibility: st.number_input(
                      label, min_value=0, max_value=120, step=1,
                      key="patient_age", help=help, label_visibility=label_visibility))

            field("Gender", "<b>Patient gender</b>.",
                  lambda label, help, label_visibility: st.selectbox(
                      label, ["Choose the gender", "Male", "Female", "Other"],
                      key="patient_gender", help=help, label_visibility=label_visibility))

            field("Income (USD/month)", "<b>Monthly income</b> before tax.",
                  lambda label, help, label_visibility: st.number_input(
                      label, min_value=0.0, step=100.0, format="%.0f",
                      key="patient_income", help=help, label_visibility=label_visibility))

            field("Employment", "<b>Employment status</b> (e.g. employed, student, retired).",
                  lambda label, help, label_visibility: st.text_input(
                      label, placeholder="e.g. employed",
                      key="patient_employment", help=help, label_visibility=label_visibility))

            field("Provider Specialty", "<b>Doctor's medical specialty</b>.",
                  lambda label, help, label_visibility: st.text_input(
                      label, placeholder="e.g. cardiology",
                      key="provider_specialty", help=help, label_visibility=label_visibility))

            field("Claim Type", "<b>Type of claim</b> (medical, dental, vision…).",
                  lambda label, help, label_visibility: st.text_input(
                      label, placeholder="e.g. medical",
                      key="claim_type", help=help, label_visibility=label_visibility))

        with c2:
            field("Date of Service", "Date of treatment / service",
                  lambda label, help, label_visibility: st.date_input(
                      label, key="date_of_service", help=help, label_visibility=label_visibility))

            field("Pre-Authorization Status", "Pre-authorization status",
                  lambda label, help, label_visibility: st.selectbox(
                      label, ["Choose an option", "Yes", "No"],
                      key="pre_auth_status", help=help, label_visibility=label_visibility))

            field("Marital Status", "<b>Marital status</b> (single, married, divorced, widowed).",
                  lambda label, help, label_visibility: st.text_input(
                      label, placeholder="e.g. single",
                      key="patient_marital", help=help, label_visibility=label_visibility))

            field("Diagnosis Code", "<b>ICD-10 diagnosis code</b> from doctor.",
                  lambda label, help, label_visibility: st.text_input(
                      label, placeholder="e.g. J18.9",
                      key="diagnosis", help=help, label_visibility=label_visibility))

            field("Procedure Code", "<b>CPT procedure code</b>.",
                  lambda label, help, label_visibility: st.text_input(
                      label, placeholder="e.g. 99213",
                      key="procedure", help=help, label_visibility=label_visibility))

            field("Submission Method", "<b>How the claim was submitted</b>.",
                  lambda label, help, label_visibility: st.text_input(
                      label, placeholder="e.g. online",
                      key="claim_submission_method", help=help, label_visibility=label_visibility))

            field("Claim Status", "<b>Current status</b> of the claim.",
                  lambda label, help, label_visibility: st.text_input(
                      label, placeholder="e.g. pending",
                      key="claim_status", help=help, label_visibility=label_visibility))

            field("Total Claim Amount (USD)",
                  "<b>Total claim amount</b> before insurance adjustments.",
                  lambda label, help, label_visibility: st.number_input(
                      label, min_value=0.0, step=100.0, format="%.0f",
                      key="claim_amount", help=help, label_visibility=label_visibility))

        c_btn = st.columns([1, 2, 1])
        with c_btn[1]:
            submitted = st.form_submit_button("Submit Claim", use_container_width=True)

    # ── Validation + API call — OUTSIDE form, INSIDE right column ───────────
    if submitted:
        st.session_state["_submitted_once"] = True
        st.session_state["_api_result"]     = None
        errors = validate()
        st.session_state["error_fields"] = errors

        if errors:
            # Không rerun — render lại ngay với highlight đỏ
            # (Streamlit sẽ re-render form ở trên với error_fields đã set)
            if len(errors) == 1:
                st.error(f"Please complete: **{errors[0]}**")
            else:
                bullet = "\n".join(f"- {e}" for e in errors)
                st.error(f"Please complete the following fields:\n{bullet}")
            # st.stop() để không chạy xuống API call
            st.stop()

        # Validation passed → call API
        payload = {
            "PatientAge":              int(st.session_state.patient_age),
            "PatientGender":           st.session_state.patient_gender or "Other",
            "PatientIncome":           float(st.session_state.patient_income),
            "PatientEmploymentStatus": str(st.session_state.patient_employment).strip() or "unknown",
            "PatientMaritalStatus":    str(st.session_state.patient_marital).strip()    or "unknown",
            "ProviderSpecialty":       str(st.session_state.provider_specialty).strip() or "unknown",
            "ClaimType":               str(st.session_state.claim_type).strip()         or "unknown",
            "ClaimAmount":             float(st.session_state.claim_amount),
            "DiagnosisCode":           str(st.session_state.diagnosis).strip()          or "UNKNOWN",
            "ProcedureCode":           str(st.session_state.procedure).strip()          or "UNKNOWN",
            "PolicyNumber":            str(st.session_state.policy_number).strip()      or "None",
            "DateOfService":           str(st.session_state.date_of_service),
            "HospitalName":            str(st.session_state.hospital_name).strip(),
            "PreAuthorizationStatus":  str(st.session_state.pre_auth_status).strip(),
            "ClaimSubmissionMethod":   str(st.session_state.claim_submission_method).strip() or "unknown",
            "ClaimStatus":             str(st.session_state.claim_status).strip()       or "pending",
        }

        with st.spinner("Processing claim…"):
            try:
                res = requests.post(API_URL, json=payload, timeout=30)
                if res.status_code != 200:
                    st.error(f"Server error: HTTP {res.status_code}")
                    st.stop()
                st.session_state["_api_result"] = res.json()
            except Exception as e:
                st.error(f"Connection error: {e}")
                st.stop()

# ── Result display — always rendered if _api_result is set ──────────────────
result = st.session_state.get("_api_result")
if result:
    decision      = result.get("decision", "Pending")
    reimbursement = result.get("reimbursement_amount")
    explanation   = result.get("explanation", "")
    confidence    = result.get("confidence")

    st.markdown("---")
    if decision == "Approved":
        st.success(f"**Claim Status: {decision}**")
    elif decision == "Denied":
        st.error(f"**Claim Status: {decision}**")
    else:
        st.warning(f"**Claim Status: {decision}**")

    col_r1, col_r2 = st.columns(2)
    with col_r1:
        amt = reimbursement if reimbursement is not None else 0.0
        st.metric("Expected Reimbursement", f"${amt:,.2f}")
    with col_r2:
        if confidence is not None:
            st.metric("Confidence", f"{confidence:.0%}")

    if explanation:
        st.info(explanation)
    elif result.get("reason"):
        st.caption(f"Reason: {result.get('reason')}")

    if result.get("flagged_items"):
        with st.expander("⚠ Flagged Items"):
            for item in result["flagged_items"]:
                st.write(f"• {item}")