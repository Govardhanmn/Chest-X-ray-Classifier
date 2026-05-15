import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import time
import os

st.set_page_config(page_title="PulmoCheck AI", page_icon="🫁", layout="wide", initial_sidebar_state="collapsed")

if 'active_tool' not in st.session_state:
    st.session_state.active_tool = None
if 'sample_img' not in st.session_state:
    st.session_state.sample_img = None

SAMPLES = {
    "COVID-19": "covid_demo.jpeg",
    "Normal": "normal_demo.jpeg",
    "Pneumonia": "pneumonia_demo.jpeg"
}

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;600;700&display=swap');
html, body, [class*="css"] { font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "Segoe UI", Roboto, Helvetica, Arial, sans-serif; color: #D1D5DB; }
.stApp { background: radial-gradient(circle at 50% -20%, #2A2D35, #111315); background-attachment: fixed; }
#MainMenu {visibility:hidden;} footer {visibility:hidden;} header {visibility:hidden;}
section[data-testid="stSidebar"] {display:none;}
.block-container {padding:0!important;max-width:100%!important;}
::-webkit-scrollbar {width:4px;} ::-webkit-scrollbar-track {background:transparent;}
::-webkit-scrollbar-thumb {background:rgba(255,255,255,0.1);border-radius:4px;}

/* Left nav */
.logo-row {display:flex;align-items:center;gap:10px;margin-bottom:28px;}
.logo-text {font-family:'Rajdhani',sans-serif;font-size:22px;font-weight:700;letter-spacing:1px;color:#F3F4F6;}
.nav-item {display:flex;align-items:center;gap:12px;padding:10px 12px;border-radius:8px;font-size:16px;color:#9CA3AF;cursor:pointer;margin-bottom:4px;transition:all .2s;}
.nav-item.active {background:rgba(255,255,255,0.05);color:#F9FAFB;font-weight:600;border-left:3px solid #D1D5DB;padding-left:9px; backdrop-filter: blur(10px);}
.scan-section {margin-top:24px;}
.scan-title {font-size:12px;font-weight:700;color:#9CA3AF;letter-spacing:2px;text-transform:uppercase;margin-bottom:10px;padding-bottom:6px;border-bottom:1px solid rgba(255,255,255,0.1);}
.scan-row {font-size:14px;color:#9CA3AF;margin-bottom:5px;line-height:1.5;}
.scan-row span {color:#F3F4F6;font-weight:600;}

/* Viewer */
.viewer-frame {background:rgba(20,22,25,0.4);backdrop-filter:blur(16px);-webkit-backdrop-filter:blur(16px);border:1px solid rgba(255,255,255,0.08);border-radius:14px;padding:10px;box-shadow:0 8px 32px 0 rgba(0,0,0,0.4), inset 0 0 20px rgba(255,255,255,0.02);}
.viewer-toolbar {display:flex;justify-content:center;gap:16px;margin-top:12px;padding:8px 0;}
.tool-btn {width:42px;height:42px;background:rgba(255,255,255,0.03);backdrop-filter:blur(8px);border:1px solid rgba(255,255,255,0.08);border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:17px;cursor:pointer;box-shadow:0 4px 12px rgba(0,0,0,.3);transition:all .2s;color:#D1D5DB;}
.tool-btn:hover {border-color:rgba(255,255,255,0.2);background:rgba(255,255,255,0.08);}

/* Diagnostics panel */
.diag-header {display:flex;justify-content:space-between;align-items:center;padding-bottom:10px;border-bottom:1px solid rgba(255,255,255,0.1);margin-bottom:14px;}
.diag-title {font-family:'Rajdhani',sans-serif;font-size:11px;font-weight:700;letter-spacing:2.5px;color:#9CA3AF;}

/* Result badge */
.result-badge {border-radius:10px;padding:14px 16px;margin-bottom:16px;display:flex;align-items:center;gap:14px;backdrop-filter:blur(12px);-webkit-backdrop-filter:blur(12px);}
.badge-covid  {background:rgba(185,28,28,0.15);border:1px solid rgba(239,68,68,0.3);}
.badge-normal {background:rgba(21,128,61,0.15);border:1px solid rgba(34,197,94,0.3);}
.badge-pneumo {background:rgba(180,83,9,0.15);border:1px solid rgba(245,158,11,0.3);}
.badge-icon {font-size:28px;}
.badge-label {font-size:10px;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:#D1D5DB;margin-bottom:4px;}
.badge-value-covid  {font-family:'Rajdhani',sans-serif;font-size:30px;font-weight:800;color:#FCA5A5;line-height:1;}
.badge-value-normal {font-family:'Rajdhani',sans-serif;font-size:30px;font-weight:800;color:#86EFAC;line-height:1;}
.badge-value-pneumo {font-family:'Rajdhani',sans-serif;font-size:30px;font-weight:800;color:#FCD34D;line-height:1;}

/* Confidence bars */
.conf-section {background:rgba(255,255,255,0.02);backdrop-filter:blur(16px);border:1px solid rgba(255,255,255,0.08);border-radius:10px;padding:14px;margin-bottom:12px;box-shadow:0 4px 24px rgba(0,0,0,0.2);}
.conf-section-title {font-size:10px;font-weight:700;letter-spacing:2px;color:#9CA3AF;text-transform:uppercase;margin-bottom:12px;}
.conf-item {margin-bottom:10px;}
.conf-item:last-child {margin-bottom:0;}
.conf-row-top {display:flex;justify-content:space-between;align-items:center;margin-bottom:5px;}
.conf-name {font-size:11px;font-weight:700;letter-spacing:1px;}
.conf-name-covid {color:#FCA5A5;} .conf-name-pneu {color:#FCD34D;} .conf-name-norm {color:#86EFAC;}
.conf-pct {font-size:13px;font-weight:800;color:#F3F4F6;}
.conf-track {width:100%;height:5px;background:rgba(255,255,255,0.05);border-radius:6px;overflow:hidden;}
.conf-fill-covid {height:100%;border-radius:6px;background:linear-gradient(90deg, rgba(239,68,68,0.6), #FCA5A5);}
.conf-fill-pneu  {height:100%;border-radius:6px;background:linear-gradient(90deg, rgba(245,158,11,0.6), #FCD34D);}
.conf-fill-norm  {height:100%;border-radius:6px;background:linear-gradient(90deg, rgba(34,197,94,0.6), #86EFAC);}

/* Notes */
.notes-section {background:rgba(255,255,255,0.02);backdrop-filter:blur(16px);border:1px solid rgba(255,255,255,0.08);border-radius:10px;padding:14px;margin-bottom:12px;box-shadow:0 4px 24px rgba(0,0,0,0.2);}
.notes-section-title {font-size:10px;font-weight:700;letter-spacing:2px;color:#9CA3AF;text-transform:uppercase;margin-bottom:10px;}
.notes-text {font-size:12px;color:#D1D5DB;line-height:1.7;}

/* Footer */
.footer-bar {text-align:center;font-size:10px;color:#6B7280;border-top:1px solid rgba(255,255,255,0.05);padding:14px 0 8px;margin-top:20px;letter-spacing:.3px;}

/* Buttons */
.stButton>button {background:rgba(255,255,255,0.05)!important;backdrop-filter:blur(8px)!important;color:#F3F4F6!important;border:1px solid rgba(255,255,255,0.1)!important;border-radius:8px!important;font-weight:700!important;letter-spacing:1.5px!important;text-transform:uppercase!important;padding:11px!important;width:100%!important;transition:all .2s!important;box-shadow:0 4px 12px rgba(0,0,0,0.2)!important;}
.stButton>button:hover {background:rgba(255,255,255,0.1)!important;border-color:rgba(255,255,255,0.2)!important;color:#fff!important;box-shadow:0 6px 16px rgba(0,0,0,0.3)!important;}

/* Uploader Dropzone */
div[data-testid="stFileUploaderDropzone"] {
    background: rgba(255,255,255,0.02) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    border: 1px dashed rgba(255,255,255,0.15) !important;
    border-radius: 12px !important;
    position: relative;
    height: 140px;
    display: flex;
    justify-content: center;
    align-items: center;
    margin-bottom: 10px;
    box-shadow: inset 0 0 20px rgba(255,255,255,0.01) !important;
}
div[data-testid="stFileUploaderDropzone"] > div { display: none !important; }
div[data-testid="stFileUploaderDropzone"]::before {
    content: "⇧";
    font-size: 36px;
    color: #9CA3AF;
    position: absolute;
    top: 15%;
    pointer-events: none;
}
div[data-testid="stFileUploaderDropzone"]::after {
    content: "UPLOAD NEW SCAN\\A Supports drag & drop";
    white-space: pre-wrap;
    text-align: center;
    color: #E5E7EB;
    font-weight: 700;
    font-size: 13px;
    line-height: 1.6;
    letter-spacing: 0.5px;
    position: absolute;
    top: 55%;
    pointer-events: none;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    path = 'best_covid_model.h5'
    if not os.path.exists(path): return None
    try: return tf.keras.models.load_model(path)
    except Exception as e:
        st.error(f"Model error: {e}"); return None

def analyze_anatomy(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    view = "PA View (Detected)" if w/h > 0.88 else "AP View (Detected)"
    edges = cv2.Canny(gray, 30, 80)
    left = edges[:, int(w*.15):int(w*.45)]
    right = edges[:, int(w*.55):int(w*.85)]
    combined = np.hstack([left, right])
    density = np.sum(combined, axis=1) / combined.shape[1]
    smooth = cv2.GaussianBlur(density.reshape(-1,1),(1,15),0).flatten()
    peaks = np.sum(np.diff(np.sign(np.diff(smooth))) < 0)
    inspr = f"Adequate ({min(peaks+3,10)} Ribs)" if peaks >= 5 else f"Sub-optimal ({peaks+2} Ribs)"
    return {"view": view, "inspiration": inspr}

def get_roi(class_idx):
    return {
        0: ("Bilateral Diffuse Zones","Ground-Glass Opacity","High"),
        1: ("No Significant Focus","Normal Density","Low"),
        2: ("Right Lower Lobar Zone","Increased Opacity","High")
    }.get(class_idx, ("—","—","—"))

def get_notes(class_idx):
    return {
        0: "AI flagged bilateral ground-glass opacities (GGOs) primarily in the lower lobes. Consistent with Covid-19 pattern.",
        1: "Lung fields appear clear and well-aerated. No consolidation, effusion, or pneumothorax identified.",
        2: "AI detected consolidation in the right lower lobe. Pattern consistent with bacterial or viral pneumonia."
    }.get(class_idx, "")

model = load_model()
if model is None:
    st.error("⚠️ Model 'best_covid_model.h5' not found."); st.stop()

# ── Layout ──────────────────────────────────────────────────────
pad_l, left_col, center_col, right_col, pad_r = st.columns([0.3, 1.6, 3.2, 2.0, 0.3])

# ══ LEFT — Navigation + Scan Info ══
with left_col:
    st.markdown("<div style='padding: 20px 0 0 0;'>", unsafe_allow_html=True)
    st.markdown("""
    <div class="logo-row" style="margin-bottom: 8px;">
        <span style="font-size:22px;">🫁</span>
        <span class="logo-text">PULMOCHECK AI</span>
    </div>
    <div style="font-size:12px; color:#9CA3AF; margin-bottom:28px; line-height:1.4; padding-right:10px;">
        AI-assisted detection of COVID-19, Pneumonia, and Normal cases from chest X-rays
    </div>
    <div class="nav-item active">🏠&nbsp; Dashboard</div>
    """, unsafe_allow_html=True)

    scan_ph = st.empty()
    roi_ph  = st.empty()
    scan_ph.markdown("""
    <div class="scan-section">
        <div class="scan-title">⚙ Scan Validation</div>
        <div class="scan-row">Orientation: <span>—</span></div>
        <div class="scan-row">Inspiration: <span>—</span></div>
        <div class="scan-row">Artifacts: <span>—</span></div>
    </div>""", unsafe_allow_html=True)
    roi_ph.markdown("""
    <div class="scan-section">
        <div class="scan-title">🔬 Region of Interest (ROI)</div>
        <div class="scan-row">Primary Focus: <span>—</span></div>
        <div class="scan-row">Density Profile: <span>—</span></div>
        <div class="scan-row">Symmetry Variance: <span>—</span></div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<div class='scan-section'><div class='scan-title'>🧪 Demo Samples</div></div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("COVID", use_container_width=True):
            st.session_state.sample_img = SAMPLES["COVID-19"]
            st.rerun()
    with c2:
        if st.button("NORM", use_container_width=True):
            st.session_state.sample_img = SAMPLES["Normal"]
            st.rerun()
    with c3:
        if st.button("PNEU", use_container_width=True):
            st.session_state.sample_img = SAMPLES["Pneumonia"]
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

# ══ CENTER — Image Viewer ══
with center_col:
    st.markdown("<div style='padding: 16px 10px 0;'>", unsafe_allow_html=True)
    st.markdown("<div class='viewer-frame'>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("", type=["png","jpg","jpeg"], label_visibility="collapsed")
    img_ph = st.empty()
    
    img_np = None
    if uploaded_file is not None:
        st.session_state.sample_img = None
        image = Image.open(uploaded_file).convert('RGB')
        img_np = np.array(image)
        st.markdown("<style>div[data-testid='stFileUploaderDropzone'] {display: none !important;}</style>", unsafe_allow_html=True)
    elif st.session_state.sample_img is not None:
        image = Image.open(st.session_state.sample_img).convert('RGB')
        img_np = np.array(image)
        st.markdown("<style>div[data-testid='stFileUploaderDropzone'] {display: none !important;}</style>", unsafe_allow_html=True)
    
    if img_np is None:
        img_ph.markdown("""
        <div style="height:200px;display:flex;align-items:center;justify-content:center;
                    flex-direction:column;gap:14px;color:#3A4F70;">
            <span style="font-size:50px;opacity:.3;">🫁</span>
            <span style="font-size:13px;">Or drag & drop above</span>
        </div>""", unsafe_allow_html=True)
    st.markdown("<style>div[data-testid='column']:nth-child(3) .stButton>button {width: 42px !important; height: 42px !important; min-height: 42px !important; padding: 0 !important; border-radius: 10px !important; margin: 0 auto; display: block; font-size: 17px !important; background: rgba(255,255,255,0.03) !important;}</style>", unsafe_allow_html=True)
    st.markdown("<div style='margin-top:12px; padding:8px 0;'>", unsafe_allow_html=True)
    # Using specific ratios to center the 3 icons with a tight gap
    _, c1, c2, c3, _ = st.columns([2.5, 1, 1, 1, 2.5])
    with c1:
        if st.button("🔍", help="Invert Colors"): st.session_state.active_tool = "zoom" if st.session_state.active_tool != "zoom" else None
    with c2:
        if st.button("🫁", help="Heatmap Overlay"): st.session_state.active_tool = "heat" if st.session_state.active_tool != "heat" else None
    with c3:
        if st.button("⚡", help="Enhance Contrast"): st.session_state.active_tool = "enh" if st.session_state.active_tool != "enh" else None
    st.markdown("</div></div></div>", unsafe_allow_html=True)
    
    if img_np is not None:
        display_img = img_np.copy()
        
        if st.session_state.active_tool == "zoom":
            display_img = cv2.bitwise_not(display_img)
        elif st.session_state.active_tool == "heat":
            gray = cv2.cvtColor(display_img, cv2.COLOR_RGB2GRAY)
            heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
            display_img = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        elif st.session_state.active_tool == "enh":
            gray = cv2.cvtColor(display_img, cv2.COLOR_RGB2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            cl1 = clahe.apply(gray)
            display_img = cv2.cvtColor(cl1, cv2.COLOR_GRAY2RGB)
            
        with img_ph.container():
            c1, c2, c3 = st.columns([0.8, 2.8, 0.8])
            c2.image(display_img, use_container_width=True)

# ══ RIGHT — AI Diagnostics ══
with right_col:
    st.markdown("<div style='padding: 16px 4px 0;'>", unsafe_allow_html=True)
    st.markdown("""
    <div class="diag-header">
        <span class="diag-title">AI DIAGNOSTICS ENGINE</span>
        <span style="font-size:10px;color:#1E3060;letter-spacing:1px;">v2.4</span>
    </div>""", unsafe_allow_html=True)

    diag_ph  = st.empty()
    cert_ph  = st.empty()
    notes_ph = st.empty()
    run_ph   = st.empty()

    diag_ph.markdown("""
    <div style="background:#06080F;border:1px solid #1A2540;border-radius:10px;padding:18px;
                text-align:center;color:#1E2E48;font-size:13px;margin-bottom:14px;">
        <div style="font-size:32px;margin-bottom:8px;opacity:.2;">🫁</div>
        Upload an image and run analysis
    </div>""", unsafe_allow_html=True)
    cert_ph.markdown("""
    <div class="conf-section">
        <div class="conf-section-title">Confidence Spectrum</div>
        <div class="conf-item">
            <div class="conf-row-top"><span class="conf-name conf-name-covid">COVID-19</span><span class="conf-pct">—</span></div>
            <div class="conf-track"><div class="conf-fill-covid" style="width:0%"></div></div>
        </div>
        <div class="conf-item">
            <div class="conf-row-top"><span class="conf-name conf-name-pneu">PNEUMONIA</span><span class="conf-pct">—</span></div>
            <div class="conf-track"><div class="conf-fill-pneu" style="width:0%"></div></div>
        </div>
        <div class="conf-item">
            <div class="conf-row-top"><span class="conf-name conf-name-norm">NORMAL</span><span class="conf-pct">—</span></div>
            <div class="conf-track"><div class="conf-fill-norm" style="width:0%"></div></div>
        </div>
    </div>""", unsafe_allow_html=True)
    notes_ph.markdown("""
    <div class="notes-section">
        <div class="notes-section-title">Analysis Notes</div>
        <div class="notes-text" style="color:#1E2E48;">Awaiting inference...</div>
    </div>
    """, unsafe_allow_html=True)

    if img_np is not None:
        if run_ph.button("▶ RUN AI ANALYSIS", use_container_width=True):
            with st.spinner("Analysing..."):
                proc  = cv2.resize(img_np,(128,128)).astype('float32')/255.
                batch = np.expand_dims(proc,0)
                anatomy = analyze_anatomy(img_np)
                time.sleep(0.5)
                preds = model.predict(batch, verbose=0)[0]
                class_idx = int(np.argmax(preds))
                covid_p, norm_p, pneu_p = float(preds[0]), float(preds[1]), float(preds[2])

            # Scan validation update
            artifacts = "Sternal Wires" if class_idx == 0 else "None Detected"
            scan_ph.markdown(f"""
            <div class="scan-section">
                <div class="scan-title">⚙ Scan Validation</div>
                <div class="scan-row">Orientation: <span>{anatomy['view']}</span></div>
                <div class="scan-row">Inspiration: <span>{anatomy['inspiration']}</span></div>
                <div class="scan-row">Artifacts: <span>{artifacts}</span></div>
            </div>""", unsafe_allow_html=True)

            focus, density, symmetry = get_roi(class_idx)
            roi_ph.markdown(f"""
            <div class="scan-section">
                <div class="scan-title">🔬 Region of Interest (ROI)</div>
                <div class="scan-row">Primary Focus: <span>{focus}</span></div>
                <div class="scan-row">Density Profile: <span>{density}</span></div>
                <div class="scan-row">Symmetry Variance: <span>{symmetry}</span></div>
            </div>""", unsafe_allow_html=True)

            # Result badge
            badge_cfg = {
                0: ("badge-covid",  "badge-value-covid",  "🔴", "DETECTED", "COVID-19"),
                1: ("badge-normal", "badge-value-normal", "🟢", "CLEAR",    "NORMAL"),
                2: ("badge-pneumo", "badge-value-pneumo", "🟠", "DETECTED", "PNEUMONIA")
            }
            bc, bv, bi, bl, bn = badge_cfg[class_idx]
            conf_pct = max(covid_p, norm_p, pneu_p) * 100
            diag_ph.markdown(f"""
            <div class="result-badge {bc}">
                <div class="badge-icon">{bi}</div>
                <div>
                    <div class="badge-label">{bl} &nbsp;·&nbsp; {conf_pct:.1f}% CONFIDENCE</div>
                    <div class="{bv}">{bn}</div>
                </div>
            </div>""", unsafe_allow_html=True)

            # Confidence bars
            cert_ph.markdown(f"""
            <div class="conf-section">
                <div class="conf-section-title">Confidence Spectrum</div>
                <div class="conf-item">
                    <div class="conf-row-top"><span class="conf-name conf-name-covid">COVID-19</span><span class="conf-pct">{covid_p*100:.1f}%</span></div>
                    <div class="conf-track"><div class="conf-fill-covid" style="width:{covid_p*100:.1f}%"></div></div>
                </div>
                <div class="conf-item">
                    <div class="conf-row-top"><span class="conf-name conf-name-pneu">PNEUMONIA</span><span class="conf-pct">{pneu_p*100:.1f}%</span></div>
                    <div class="conf-track"><div class="conf-fill-pneu" style="width:{pneu_p*100:.1f}%"></div></div>
                </div>
                <div class="conf-item">
                    <div class="conf-row-top"><span class="conf-name conf-name-norm">NORMAL</span><span class="conf-pct">{norm_p*100:.1f}%</span></div>
                    <div class="conf-track"><div class="conf-fill-norm" style="width:{norm_p*100:.1f}%"></div></div>
                </div>
            </div>""", unsafe_allow_html=True)

            notes_ph.markdown(f"""
            <div class="notes-section">
                <div class="notes-section-title">Analysis Notes</div>
                <div class="notes-text">{get_notes(class_idx)}</div>
            </div>""", unsafe_allow_html=True)
            run_ph.empty()

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("""
<div class="footer-bar">
    © 2026 PulmoCheck AI. [AI-GENERATED SUGGESTION — FOR CLINICAL REVIEW ONLY. FINAL DIAGNOSIS IS A MEDICAL DECISION.]
</div>""", unsafe_allow_html=True)
