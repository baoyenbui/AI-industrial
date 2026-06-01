import streamlit as st
import requests
from PIL import Image
import io
import re
import unicodedata
from datetime import datetime
import time


API_URL = "http://127.0.0.1:8000/predict"
OCR_API_URL = "http://127.0.0.1:8000/ocr-claim"


st.set_page_config(page_title="Health Claim AI", layout="wide")


st.markdown("""
<style>
.stApp { background-color: white !important; }
div.block-container { background-color: white !important; padding: 2rem !important; max-width: 1200px; }
section { background-color: transparent !important; }
div[data-testid="stFileUploader"],
div[data-testid="stForm"],
div[data-testid="stExpander"] { background-color: #E6F0FF !important; border-radius: 12px; padding: 12px; }
input:not([type="hidden"]), textarea {
    background-color: white !important; color: black !important;
    border: 1px solid #B0B0B0 !important; border-radius: 8px !important;
}
h1,h2,h3,h4,h5,h6,p,span,label { color: black !important; }
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
    opacity:0 !important; position:absolute !important;
    pointer-events:none !important; height:0 !important; width:0 !important;
}
ul[role="listbox"] { background-color: white !important; border: 1px solid #B0B0B0 !important; }
li[role="option"] { background-color: white !important; color: black !important; }
li[role="option"]:hover { background-color: #E6F0FF !important; }
div[data-testid="stFormSubmitButton"] { display:flex !important; justify-content:center !important; margin-top:12px; }
div[data-testid="stFormSubmitButton"] button {
    width:180px !important; border:2px solid black !important;
    background:white !important; color:black !important; border-radius:10px !important; font-weight:600;
}
div[data-testid="stFormSubmitButton"] button:hover { background-color:#f5f5f5 !important; }
div[data-testid="stButton"] button {
    width:180px !important; background:white !important; color:black !important;
    border:2px solid black !important; border-radius:10px !important;
}
div[data-testid="stFileUploader"] button {
    background:white !important; color:black !important;
    border:2px solid black !important; border-radius:10px !important; box-shadow:none !important;
}
div[data-testid="stNumberInput"] { border:none !important; background:transparent !important; padding:0 !important; }
div[data-testid="stNumberInput"] > div > div {
    background:white !important; border:1px solid #B0B0B0 !important;
    border-radius:8px !important; overflow:hidden !important;
}
div[data-testid="stNumberInput"] input { background:white !important; border:none !important; color:black !important; }
div[data-testid="stNumberInput"] button { background:white !important; border-left:1px solid #B0B0B0 !important; }
input::placeholder, textarea::placeholder { color:rgba(0,0,0,0.35) !important; }


.fl-wrap {
    display: flex;
    align-items: center;
    gap: 6px;
    margin-bottom: 4px;
    font-weight: 600;
    line-height: 1.6;
}
.fl-wrap .fl-text {
    font-size: 14px;
    font-weight: 600;
}
.fl-wrap .fl-text.err {
    color: #CC0000 !important;
}
.fl-wrap .fl-text.ok {
    color: #111111 !important;
}
.tip-icon {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 15px;
    height: 15px;
    border-radius: 50%;
    font-size: 10px;
    font-weight: 700;
    cursor: help;
    flex-shrink: 0;
    position: relative;
    border: 1px solid #999;
    color: #999 !important;
    text-decoration: none;
}
.tip-icon.err {
    border-color: #CC0000 !important;
    color: #CC0000 !important;
}
.tip-icon .tip-box {
    display: none;
    position: absolute;
    left: 20px;
    top: -4px;
    z-index: 9999;
    background: white;
    border: 1px solid #CCC;
    border-radius: 8px;
    padding: 9px 12px;
    width: 220px;
    font-size: 12px;
    font-weight: 400;
    line-height: 1.6;
    box-shadow: 0 4px 16px rgba(0,0,0,0.12);
    pointer-events: none;
    white-space: normal;
    color: #222 !important;
}
.tip-icon:hover .tip-box { display: block; }


.result-panel {
    border-radius: 12px;
    padding: 24px;
    margin: 20px 0;
    border: 1.5px solid #ddd;
    background: white;
}
</style>
""", unsafe_allow_html=True)


st.markdown("<h1 style='margin-bottom:10px;'>Health Claim Support System</h1>", unsafe_allow_html=True)

def clean_text(x):
    if x is None: return ""
    x = unicodedata.normalize("NFKC", str(x))
    x = re.sub(r"[\u200b-\u200f\ufeff\u00a0\x00-\x1f\x7f]", "", x)
    return re.sub(r"\s+", " ", x).strip()

def safe_int(x):
    try: return int(re.search(r"\d+", str(x)).group())
    except: return 0

def safe_float(x):
    try: return float(str(x).replace(",", "."))
    except: return 0.0

def normalize_gender(x):
    xl = clean_text(x).lower()
    if "female" in xl: return "Female"
    if "male" in xl: return "Male"
    return ""

def _is_blank(v, min_len=1):
    if v is None: return True
    return str(v).strip() == "" or len(str(v).strip()) < min_len

def merge_ocr_results(results):
    SUM, FIRST = {"ClaimAmount", "PatientIncome"}, {"PatientAge"}
    merged = {}
    for data in results:
        for k, v in data.items():
            if k in SUM:
                merged[k] = merged.get(k, 0.0) + safe_float(v)
            elif k in FIRST:
                if not merged.get(k):
                    n = safe_int(v)
                    if n: merged[k] = n
            else:
                if not clean_text(merged.get(k, "")) and clean_text(v):
                    merged[k] = clean_text(v)
    return merged

DEFAULTS = {
    "patient_age": 0,
    "patient_gender": "Choose the gender",
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
    "policy_number": "",
    "date_of_service": datetime.today().date(),
    "provider_name": "",
    "pre_auth_status": "Choose an option",
    "error_fields": [],
    "_api_result": None,
    "_uploaded_images": [],
}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)

def validate():
    s, e = st.session_state, []
    if s.patient_age <= 0: e.append("Age")
    if s.patient_gender in ("Choose the gender", "", None): e.append("Gender")
    if _is_blank(s.patient_employment): e.append("Employment")
    if s.claim_amount <= 0: e.append("Total Claim Amount (USD)")
    if _is_blank(s.diagnosis): e.append("Diagnosis Code")
    if _is_blank(s.provider_name, 3): e.append("Provider")
    if s.pre_auth_status in ("Choose an option", "", None): e.append("Pre-Authorization Status")
    if _is_blank(s.procedure): e.append("Procedure Code")
    if _is_blank(s.claim_submission_method): e.append("Submission Method")
    if _is_blank(s.provider_specialty): e.append("Provider Specialty")
    if _is_blank(s.claim_type): e.append("Claim Type")
    if _is_blank(s.patient_marital): e.append("Marital Status")
    if _is_blank(s.policy_number): e.append("Policy Number / Member ID")
    return e

def fl(label, tip, hl):
    is_err = label in hl
    text_cls = "err" if is_err else "ok"
    icon_cls = "tip-icon err" if is_err else "tip-icon"
    star = "&nbsp;<span style='color:#CC0000;font-weight:700'>*</span>" if is_err else ""
    st.markdown(
        f"<div class='fl-wrap'>"
        f"<span class='fl-text {text_cls}'>{label}{star}</span>"
        f"<span class='{icon_cls}'>?<span class='tip-box'>{tip}</span></span>"
        f"</div>",
        unsafe_allow_html=True,
    )

def render_result(r):
    decision = r.get("decision", "Pending")
    reimb = r.get("reimbursement_amount") or 0.0
    baseline = r.get("baseline_amount") or 0.0
    conf = r.get("confidence") or 0.0
    expl = r.get("explanation", "")

    cls = {"Approved": "result-approved", "Denied": "result-denied"}.get(decision, "result-pending")

    meta_parts = []
    if r.get("policy_number"):
        meta_parts.append(f"Policy: <b>{r.get('policy_number')}</b>")
    if r.get("provider_name"):
        meta_parts.append(f"Provider: <b>{r.get('provider_name')}</b>")
    if r.get("date_of_service"):
        meta_parts.append(f"Date: <b>{r.get('date_of_service')}</b>")
    if r.get("pre_auth"):
        meta_parts.append(f"Pre-Auth: <b>{r.get('pre_auth')}</b>")

    meta = "  ·  ".join(meta_parts)

    full_html = f"""
    <div class='result-panel {cls}'>
        <div class='result-title'>Claim {decision}</div>
        
        <div class='result-row'>
            <div class='result-kv'>
                <span class='result-kv-label'>Reimbursement</span>
                <span class='result-kv-value'>${reimb:,.2f}</span>
            </div>
            <div class='result-kv'>
                <span class='result-kv-label'>Baseline</span>
                <span class='result-kv-value'>${baseline:,.2f}</span>
            </div>
            <div class='result-kv'>
                <span class='result-kv-label'>Confidence</span>
                <span class='result-kv-value'>{conf:.0%}</span>
            </div>
        </div>

        <div class='result-meta'>{meta}</div>
        
        <div class='result-explanation'>
            {expl}
        </div>

        <div class='iais-note'>
            <span class='info-icon'>i</span>
            This calculation follows the <a href="https://www.iais.org/">International Association of Insurance Supervisors (IAIS)</a> standard insurance methodologies for transparency and policy clarity.
        </div>
    </div>

    <style>
        .result-panel {{
            border-radius: 12px !important; 
            padding: 24px !important; 
            margin: 20px 0 !important; 
            border: 1px solid #ddd !important;
            background: white;
        }}
        
        .result-approved {{
            background: #d4edda !important;     
            border-color: #d4edda !important;
        }}
        
        .result-denied {{
            background: #fef2f2 !important;
            border-color: #f87171 !important;
        }}
        
        .result-pending {{
            background: #fffbeb !important;
            border-color: #fbbf24 !important;
        }}

        .result-title {{
            font-size:25px; 
            font-weight:700; 
            margin-bottom:18px;
        }}
        .result-approved .result-title {{ color:#166534; }}

        .result-row      {{ display:flex; gap:32px; margin:16px 0; flex-wrap:wrap; }}
        .result-kv       {{ display:flex; flex-direction:column; }}
        .result-kv-label {{ font-size:11px; color:#666; text-transform:uppercase; }}
        .result-kv-value {{ font-size:21px; font-weight:700; }}
        .result-meta     {{ font-size:13.5px; color:#444; margin:12px 0 20px; }}

        .result-explanation {{ 
            background:#fafafa; 
            padding:26px; 
            border-radius:12px; 
            border:1px solid #e5e5e5; 
        }}

        .iais-note {{
            margin-top: 24px;
            padding: 10px 14px;
            background: #f8f9fa;
            border-radius: 8px;
            font-size: 13.2px;
            color: #555;
            border-left: 3px solid #555;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .iais-note .info-icon {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 19px;
            height: 19px;
            border: 2px solid #555;
            color: #555;
            font-size: 13px;
            font-weight: 700;
            border-radius: 50%;
            flex-shrink: 0;
        }}
        .iais-note a {{
            color: #1e40af;
            text-decoration: underline;
        }}
        .iais-note a:hover {{
            color: #1e3a8a;
        }}

        .exp-container {{ max-width:100%; }}
        .exp-header {{ margin-bottom:24px; }}
        .exp-headline {{ font-size:18.5px; font-weight:700; margin-bottom:10px; }}
        .exp-sub {{ color:#555; margin-bottom:22px; }}

        .exp-amounts, .exp-section {{ 
            margin-bottom:24px; 
            padding:20px; 
            background:white; 
            border-radius:10px; 
            border:1px solid #eee;
        }}
        .exp-section-title {{
            font-size:16px; 
            font-weight:600; 
            margin-bottom:16px; 
            color:#1a3c5e;
            border-bottom:1px solid #eee;
            padding-bottom:12px;
        }}
        .exp-amount-row {{
            display:flex; 
            justify-content:space-between; 
            padding:12px 0; 
            border-bottom:1px solid #f0f0f0;
        }}
        
        .exp-amount-row:last-child {{ border-bottom:none; }}
        .exp-amount-row.covered {{ color:#1a7a3e; font-weight:600; }}
        .exp-amount-row.owe {{ color:#b91c1c; font-weight:600; }}

        .exp-calc-table {{
            background: #f8f9fa;
            border-radius: 8px;
            padding: 20px;
            max-width: 100%;
            margin: 16px 0;
        }}

        .exp-calc-row {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 0;
            border-bottom: 1px solid #e0e0e0;
            font-size: 15px;
            line-height: 1.4;
        }}

        .exp-calc-row:last-child {{
            border-bottom: none;
        }}

        .exp-calc-header {{
            font-weight: 700;
            font-size: 16px;
            color: #333;
            padding-bottom: 14px;
            border-bottom: 2px solid #ddd;
            margin-bottom: 4px;
        }}

        .exp-calc-subtle {{
            color: #555;
        }}

        .exp-calc-deduction {{
            color: #d32f2f;
        }}

        .exp-calc-divider {{
            font-weight: 600;
            color: #333;
            border-top: 2px solid #ddd;
            padding-top: 14px;
            margin-top: 6px;
        }}

        .exp-calc-total {{
            font-weight: 700;
            font-size: 17px;
            color: #2e7d32;
            background: #e8f5e9;
            padding: 14px;
            border-radius: 6px;
            margin-top: 10px;
            border: none;
        }}

        .exp-calc-label {{
            flex: 1;
            padding-right: 12px;
            color: #444;
            font-weight: 400;
        }}

        .exp-calc-value {{
            font-weight: 700;
            color: #222;
            white-space: nowrap;
            text-align: right;
            min-width: 100px;
        }}

        .exp-explanation-box {{
            background: #fff8e1;
            border-left: 4px solid #ffc107;
            padding: 16px;
            margin-top: 12px;
            border-radius: 4px;
            font-size: 0.95em;
            line-height: 1.6;
        }}

        .exp-explanation-box p {{
            margin: 8px 0;
        }}

        .exp-explanation-box p:first-child {{
            margin-top: 0;
        }}

        .exp-explanation-box p:last-child {{
            margin-bottom: 0;
        }}

        .exp-adjustment-list {{
            margin: 10px 0;
            padding-left: 20px;
        }}

        .exp-adjustment-list li {{
            margin: 6px 0;
        }}

        .exp-note {{
            background: #e3f2fd;
            padding: 10px;
            border-radius: 4px;
            margin-top: 10px;
            font-weight: 500;
        }}

        .exp-note strong {{
            color: #1976d2;
        }}

        .exp-factor-row {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 0;
            border-bottom: 1px solid #f0f0f0;
        }}
        .exp-factor-row:last-child {{
            border-bottom: none;
        }}
        .exp-factor-label {{
            flex: 1;
            padding-right: 12px;
            line-height: 1.4;
        }}
        
        .exp-badge {{
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 13px;
            font-weight: 600;
            white-space: nowrap;
        }}
        
        .exp-badge-favorable {{
            background: #d4edda;
            color: #155724;
        }}
        
        .exp-badge-reviewed {{
            background: #fff3cd;
            color: #856404;
        }}

        .exp-knowledge-text {{
            font-size: 0.95em;
            color: #555;
            line-height: 1.6;
        }}
    </style>
    """

    st.components.v1.html(full_html, height=1150, scrolling=True)


left, right = st.columns([1, 1.2])

with left:
    st.markdown(
        """
        <div style="
            border: 1px solid #e5e7eb;
            border-radius: 14px;
            background: #f9fafb;
            padding: 16px 16px 14px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.04);
        ">
            <h3 style="margin: 0 0 8px 0;">Upload Documents</h3>
        <style>
        .subtitle-box {
            background: #f8f9fa;
            border-left: 4px solid #667eea;
            padding: 16px 20px;
            margin: 0 0 30px 0;
            border-radius: 8px;
            font-style: italic;
            font-size: 14.5px;
            line-height: 1.6;
            color: #555555;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        }
        </style>
        <div class='subtitle-box'>
            Upload any relevant documents (e.g., prescriptions, insurance policy terms, or medical bills) to ensure the most accurate response. 
        </div>
        """, 
        unsafe_allow_html=True,
        )

    uploaded_files = st.file_uploader(
        "Select Images",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        st.session_state["_uploaded_images"] = [{"bytes": f.getvalue(), "name": f.name} for f in uploaded_files]
    else:
        st.session_state["_uploaded_images"] = []

    images_data = st.session_state.get("_uploaded_images", [])

    if images_data:
        n = len(images_data)
        thumb_width = 160 if n <= 2 else (120 if n <= 4 else 95)
        st.markdown(
            f"<div style='text-align:center;font-size:13px;color:#555;margin:12px 0 16px;'>{n} image{'s' if n > 1 else ''} selected</div>",
            unsafe_allow_html=True,
        )
        cols = st.columns(min(n, 5))
        for idx, img_info in enumerate(images_data):
            with cols[idx % min(n, 5)]:
                try:
                    img = Image.open(io.BytesIO(img_info["bytes"]))
                    h = int(thumb_width * img.height / img.width)
                    st.image(img.resize((thumb_width, h), Image.LANCZOS), width=thumb_width, caption=img_info["name"][:14])
                except:
                    st.error(f"Cannot load {img_info['name']}")

        _, mid, _ = st.columns([1, 2, 1])
        with mid:
            run_ocr = st.button(f"Run OCR ({n} image{'s' if n > 1 else ''})", use_container_width=True)

        if run_ocr:
            extracted, errs = [], []
            bar = st.progress(0, text="Running OCR…")
            for i, img_info in enumerate(images_data):
                bar.progress(i / n, text=f"Processing {img_info['name']} ({i+1}/{n})…")
                try:
                    r = requests.post(OCR_API_URL, files={"file": (img_info["name"], img_info["bytes"])}, timeout=30)
                    if r.status_code == 200:
                        extracted.append(r.json().get("extracted_data", {}))
                    else:
                        errs.append(f"{img_info['name']}: HTTP {r.status_code}")
                except Exception as ex:
                    errs.append(f"{img_info['name']}: {ex}")
            bar.progress(1.0, text="Done!")
            for err in errs:
                st.warning(f"OCR error — {err}")

            if extracted:
                m = merge_ocr_results(extracted)
                st.session_state.update({
                    "patient_age": safe_int(m.get("PatientAge")),
                    "patient_gender": normalize_gender(m.get("PatientGender")) or "Choose the gender",
                    "patient_income": safe_float(m.get("PatientIncome")),
                    "patient_employment": clean_text(m.get("PatientEmploymentStatus")),
                    "patient_marital": clean_text(m.get("PatientMaritalStatus")),
                    "provider_specialty": clean_text(m.get("ProviderSpecialty")),
                    "claim_type": clean_text(m.get("ClaimType")),
                    "claim_submission_method": clean_text(m.get("ClaimSubmissionMethod")),
                    "diagnosis": clean_text(m.get("DiagnosisCode")),
                    "procedure": clean_text(m.get("ProcedureCode")),
                    "claim_status": clean_text(m.get("ClaimStatus")),
                    "claim_amount": safe_float(m.get("ClaimAmount")),
                    "policy_number": clean_text(m.get("PolicyNumber")),
                    "provider_name": clean_text(m.get("ProviderName")),
                })
                pre = clean_text(m.get("PreAuthorizationStatus"))
                if pre in ("Yes", "No"):
                    st.session_state["pre_auth_status"] = pre
                date_str = m.get("DateOfService")
                if date_str:
                    for fmt in ("%d %B %Y", "%B %d %Y", "%d/%m/%Y", "%m/%d/%Y", "%Y-%m-%d", "%d-%m-%Y"):
                        try:
                            st.session_state["date_of_service"] = datetime.strptime(str(date_str).strip(), fmt).date()
                            break
                        except:
                            continue
                st.session_state["error_fields"] = []
                st.session_state["_api_result"] = None
                time.sleep(0.1)
                st.rerun()
    else:
        st.markdown(
            "<div style='text-align:center;color:#888;font-size:14.5px;margin:50px 0;'><em>No images uploaded yet.</em></div>",
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)
    
with right:
    st.markdown(
    """
    <div style="
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        background: #f9fafb;
        padding: 16px 16px 14px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    ">
        <h3 style="margin: 0 0 8px 0;">Claim Information</h3>
        <style>
        .subtitle-box {
            background: #f8f9fa;
            border-left: 4px solid #667eea;
            padding: 16px 20px;
            margin: 0 0 30px 0;
            border-radius: 8px;
            font-style: italic;
            font-size: 14.5px;
            line-height: 1.6;
            color: #555555;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        }
        </style>
        <div class='subtitle-box'>
            The extracted information may contain missing or incorrect details, so please review it carefully and complete any blank fields in the form.
        </div>      
    """,
    unsafe_allow_html=True,
    )

    hl = set(st.session_state.get("error_fields", []))

    if hl:
        st.markdown(
            "<div style='background:#fff0f0;border:1.5px solid #CC0000;border-radius:8px;"
            "padding:10px 14px;margin-bottom:14px;font-size:13px;color:#CC0000 !important;'>"
            "⚠ Please complete the required fields</div>",
            unsafe_allow_html=True,
        )

    with st.form("claim_form", clear_on_submit=False):
        c1, c2 = st.columns(2)

        with c1:
            fl("Policy Number / Member ID",
               "Your insurance policy number or member ID - usually found on your insurance card or welcome letter.",
               hl)
            st.text_input("Policy Number / Member ID", placeholder="e.g. POL123456789",
                          key="policy_number", label_visibility="collapsed")

            fl("Provider",
               "The entity that provided this bill/document (hospital, clinic, or insurance company).",
               hl)
            st.text_input("Provider", placeholder="e.g. Vinmec",
                          key="provider_name", label_visibility="collapsed")

            fl("Age",
               "The patient's current age in years. Must be between 1 and 120.",
               hl)
            st.number_input("Age", min_value=0, max_value=120, step=1,
                            key="patient_age", label_visibility="collapsed")

            fl("Gender",
               "The patient's gender as recorded in their medical or insurance record.",
               hl)
            st.selectbox("Gender", ["Choose the gender", "Male", "Female", "Other"],
                         key="patient_gender", label_visibility="collapsed")

            fl("Income (USD/month)",
               "The patient's average monthly income before tax. This helps assess the appropriate coverage tier.",
               hl)
            st.number_input("Income (USD/month)", min_value=0.0, step=100.0, format="%.0f",
                            key="patient_income", label_visibility="collapsed")

            fl("Employment",
               "Current employment status - e.g. Employed, Self-employed, Unemployed, Retired, or Student.",
               hl)
            st.text_input("Employment", placeholder="e.g. Employed",
                          key="patient_employment", label_visibility="collapsed")

            fl("Provider Specialty",
               "The medical specialty of the treating doctor - e.g. Cardiology, Orthopedics, Pediatrics.",
               hl)
            st.text_input("Provider Specialty", placeholder="e.g. Cardiology",
                          key="provider_specialty", label_visibility="collapsed")

            fl("Claim Type",
               "The category of this claim - e.g. Medical, Dental, Vision, Pharmacy, or Mental Health.",
               hl)
            st.text_input("Claim Type", placeholder="e.g. Medical",
                          key="claim_type", label_visibility="collapsed")

        with c2:
            fl("Date of Service",
               "The date when the medical treatment or service was received. Must match your medical bill.",
               hl)
            st.date_input("Date of Service", key="date_of_service", label_visibility="collapsed")

            fl("Pre-Authorization Status",
               "Whether your insurer approved this treatment in advance. Select Yes if you received a pre-auth number before the visit.",
               hl)
            st.selectbox("Pre-Authorization Status", ["Choose an option", "Yes", "No"],
                         key="pre_auth_status", label_visibility="collapsed")

            fl("Marital Status",
               "Patient's marital status - e.g. Single, Married, Divorced, or Widowed.",
               hl)
            st.text_input("Marital Status", placeholder="e.g. Single",
                          key="patient_marital", label_visibility="collapsed")

            fl("Diagnosis Code",
               "The ICD-10 code your doctor assigned - a letter followed by numbers, e.g. J18.9 (pneumonia).",
               hl)
            st.text_input("Diagnosis Code", placeholder="e.g. J18.9",
                          key="diagnosis", label_visibility="collapsed")

            fl("Procedure Code",
               "The CPT code for the treatment performed: a 5-digit number on your bill, e.g. 99213.",
               hl)
            st.text_input("Procedure Code", placeholder="e.g. 99213",
                          key="procedure", label_visibility="collapsed")

            fl("Submission Method",
               "How this claim is being submitted - e.g. Online, Paper, Fax, or via the Hospital directly.",
               hl)
            st.text_input("Submission Method", placeholder="e.g. Online",
                          key="claim_submission_method", label_visibility="collapsed")

            fl("Claim Status",
               "Current status of this claim if known — e.g. Pending, Approved, or Denied.",
               hl)
            st.text_input("Claim Status", placeholder="e.g. Pending",
                          key="claim_status", label_visibility="collapsed")

            fl("Total Claim Amount (USD)",
               "Total amount billed by the provider before insurance adjustment — found at the bottom of your invoice.",
               hl)
            st.number_input("Total Claim Amount (USD)", min_value=0.0, step=100.0, format="%.0f",
                            key="claim_amount", label_visibility="collapsed")

        submitted = st.form_submit_button("Submit Claim", use_container_width=True)

if submitted or st.session_state.get("_api_result"):
    if submitted:
        errors = validate()
        if errors:
            st.session_state["error_fields"] = errors
            time.sleep(0.1)
            st.rerun()
        else:
            st.session_state["error_fields"] = []

            payload = {
                "PatientAge": int(st.session_state.patient_age),
                "PatientGender": st.session_state.patient_gender or "Other",
                "PatientIncome": float(st.session_state.patient_income),
                "PatientEmploymentStatus": str(st.session_state.patient_employment).strip() or "unknown",
                "PatientMaritalStatus": str(st.session_state.patient_marital).strip() or "unknown",
                "ProviderSpecialty": str(st.session_state.provider_specialty).strip() or "unknown",
                "ClaimType": str(st.session_state.claim_type).strip() or "unknown",
                "ClaimAmount": float(st.session_state.claim_amount),
                "DiagnosisCode": str(st.session_state.diagnosis).strip() or "UNKNOWN",
                "ProcedureCode": str(st.session_state.procedure).strip() or "UNKNOWN",
                "PolicyNumber": str(st.session_state.policy_number).strip() or "",
                "DateOfService": str(st.session_state.date_of_service),
                "ProviderName": str(st.session_state.provider_name).strip(),
                "PreAuthorizationStatus": str(st.session_state.pre_auth_status).strip(),
                "ClaimSubmissionMethod": str(st.session_state.claim_submission_method).strip() or "unknown",
                "ClaimStatus": str(st.session_state.claim_status).strip() or "pending",
            }

            with st.spinner("Processing claim..."):
                try:
                    r = requests.post(API_URL, json=payload, timeout=30)
                    if r.status_code == 200:
                        st.session_state["_api_result"] = r.json()
                        time.sleep(0.1)
                        st.rerun()
                    else:
                        st.error(f"Server error: HTTP {r.status_code}")
                except Exception as ex:
                    st.error(f"Connection error: {ex}")

if st.session_state.get("_api_result"):
    render_result(st.session_state["_api_result"])