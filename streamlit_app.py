import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(page_title="DFU Wound-Staging Prototype", page_icon="🦶", layout="wide")


@st.cache_resource(show_spinner="Loading the MobileNetV3 model…")
def load_core():
    import dfu_core  # loads the model once per server, not on every click
    return dfu_core


core = load_core()

st.title("🏥 JSED 2026: Automated Diabetic Foot Wound-Staging & SINBAD Decision-Support Prototype")
st.warning(
    "**Research prototype only — not a medical device.** Tissue percentages, the SINBAD score and the "
    "management suggestions come from an unvalidated colour heuristic applied to a Grad-CAM map. "
    "Do not use them to guide patient care.",
    icon="⚠️",
)
st.markdown(
    "**1.** Detects ulcer (MobileNetV3-Large) → **2.** Quantifies red/yellow/black tissue % → "
    "**3.** Calculates the IWGDF SINBAD score and suggested debridement/antibiotic actions"
)

left, right = st.columns(2, gap="large")

with left:
    upload = st.file_uploader("Upload foot photograph", type=["jpg", "jpeg", "png", "webp", "bmp"])
    if upload is not None:
        st.image(upload, caption="Uploaded photo", use_container_width=True)
    site = st.radio(
        "SINBAD 1: Anatomical Site",
        ["Forefoot (Toes/Metatarsal Heads)", "Midfoot/Hindfoot (Heel/Arch)"],
    )
    ischemia = st.radio(
        "SINBAD 2: Ischemia Status",
        ["Intact Pedal Blood Flow (Palpable Pulses)", "Reduced/Absent Pedal Pulses (Suspected PAD)"],
    )
    neuropathy = st.radio(
        "SINBAD 3: Peripheral Neuropathy",
        ["Protective Sensation Intact", "Neuropathy Present (10g Monofilament Loss)"],
        index=1,
    )
    depth = st.radio(
        "SINBAD 6: Ulcer Depth (Probe-to-Bone)",
        ["Superficial (Confined to Skin/Subcutaneous)", "Deep (Reaching Muscle, Tendon, or Bone)"],
    )
    tele = st.checkbox("Simulate Poor Patient Smartphone Photo (Blur + Low Lighting)")
    run = st.button("Run Automated Wound Staging & SINBAD Consult", type="primary", use_container_width=True)

with right:
    if run and upload is None:
        st.info("Please upload a foot photograph first.")
    elif run:
        image = np.array(Image.open(upload).convert("RGB"))
        with st.spinner("Running the model and Grad-CAM…"):
            cam_img, seg_img, tissue, report = core.digital_podiatry_consult(
                image, site, ischemia, neuropathy, depth, tele
            )
        c1, c2 = st.columns(2)
        c1.image(cam_img, caption="1. AI Ulcer Localization (Grad-CAM)", use_container_width=True)
        c2.image(seg_img, caption="2. Tissue Segmentation (Red/Yellow/Black + Cyan Margin)", use_container_width=True)
        st.subheader("3. Wound-Bed Tissue Composition")
        for name, value in tissue.items():
            st.progress(min(max(float(value), 0.0), 1.0), text=f"{name}: {float(value) * 100:.1f}%")
        st.subheader("4. SINBAD Score & Suggested Actions")
        st.code(report, language=None, wrap_lines=True)
    else:
        st.caption("Upload a photo, fill in the SINBAD fields, then press **Run**. Results appear here.")
