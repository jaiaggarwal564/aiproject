# app/streamlit_app.py
import streamlit as st
import joblib
import json
import pandas as pd
import numpy as np
import base64
from pathlib import Path
from streamlit.components.v1 import html

# -----------------------------
# Config
# -----------------------------
MODEL_PATH = "D:/protein_solubility_project/models/xgb2_solubility.joblib"
FEATURE_COLS_PATH = "D:/protein_solubility_project/models/feature_cols.json"

st.set_page_config(page_title="Protein Solubility — Demo", layout="centered", initial_sidebar_state="collapsed")

# -----------------------------
# Styles & Background + Glass card
# -----------------------------
PAGE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="stApp"] {
  height: 100%;
  background: radial-gradient(circle at 10% 10%, #0f172a 0%, #071029 25%, #02101b 100%);
  font-family: "Inter", sans-serif;
}

/* blurred animated container */
.background-hero {
  position: relative;
  width: 100%;
  height: 380px;
  border-radius: 16px;
  overflow: hidden;
  margin-bottom: 18px;
  box-shadow: 0 8px 40px rgba(2,6,23,0.7);
}

/* translucent card */
.card {
  background: rgba(255,255,255,0.06);
  backdrop-filter: blur(8px) saturate(120%);
  border-radius: 14px;
  padding: 18px;
  box-shadow: 0 6px 30px rgba(2,6,23,0.5);
  color: #e6eef8;
}

/* main title */
.title {
  font-size: 20px;
  font-weight: 700;
  margin-bottom: 6px;
  color: #e6eef8;
}

/* small notes */
.small {
  color: #9fb3d6;
  font-size: 12px;
}

/* input area tweaks */
textarea[role="textbox"] {
  font-family: monospace;
  font-size: 13px;
}

/* footer */
.footer {
  color: #9fb3d6;
  font-size: 12px;
  margin-top: 16px;
}
</style>
"""

st.markdown(PAGE_CSS, unsafe_allow_html=True)

# -----------------------------
# Load model & columns
# -----------------------------
@st.cache_resource
def load_model_and_cols():
    model = joblib.load(MODEL_PATH)
    with open(FEATURE_COLS_PATH, "r") as f:
        cols = json.load(f)
    return model, cols

try:
    model, FEATURE_COLS = load_model_and_cols()
except Exception as e:
    st.error(f"Model loading error: {e}")
    st.stop()

# -----------------------------
# Helpers (same as notebook)
# -----------------------------
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
hydropathy = {
    'A': 1.8,  'C': 2.5,  'D': -3.5, 'E': -3.5, 'F': 2.8,
    'G': -0.4, 'H': -3.2, 'I': 4.5,  'K': -3.9, 'L': 3.8,
    'M': 1.9,  'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
    'S': -0.8, 'T': -0.7, 'V': 4.2,  'W': -0.9, 'Y': -1.3
}

def aa_composition(seq):
    seq = seq.strip().upper()
    counts = {aa: 0 for aa in AMINO_ACIDS}
    total = 0
    for aa in seq:
        if aa in counts:
            counts[aa] += 1
            total += 1
    if total == 0:
        return {aa: 0.0 for aa in AMINO_ACIDS}
    return {aa: counts[aa] / total for aa in AMINO_ACIDS}

def compute_physchem(seq):
    seq = seq.strip().upper()
    length = len(seq)
    if length == 0:
        return {"seq_length": 0, "hydrophobicity": 0.0, "aromatic_fraction": 0.0}
    hyd = sum(hydropathy.get(aa,0) for aa in seq) / length
    aromatic = sum(seq.count(aa) for aa in "FWY") / length
    return {"seq_length": length, "hydrophobicity": hyd, "aromatic_fraction": aromatic}

def build_feature_vector(seq):
    aa_feats = aa_composition(seq)
    phys = compute_physchem(seq)
    row = []
    for c in FEATURE_COLS:
        if c in aa_feats:
            row.append(aa_feats[c])
        elif c in phys:
            row.append(phys[c])
        else:
            row.append(0.0)
    return np.array(row).reshape(1, -1)

# -----------------------------
# Layout: sidebar navigation
# -----------------------------
page = st.sidebar.selectbox("Navigation", ["Home", "Project Details", "Samples / Download"], index=0)

# -----------------------------
# HERO with 3D viewer (NGL) embedded
# -----------------------------
st.markdown('<div class="background-hero card">', unsafe_allow_html=True)

# Use NGL viewer via CDN. This will fetch PDB by id from RCSB (internet required).
# You can change pdbId to any PDB (e.g., 1EMA, 1GFL, 1CRN)
pdbId = "1EMA"  # example; change if you like

ngl_html = f"""
<div id="viewport" style="width:100%; height:380px;"></div>
<script src="https://unpkg.com/ngl@0.10.4/dist/ngl.js"></script>
<script>
const stage = new NGL.Stage("viewport");
fetch("https://files.rcsb.org/download/{pdbId}.pdb")
  .then(resp => resp.text())
  .then(data => {{
    const blob = new Blob([data], {{type: "chemical/x-pdb"}});
    stage.loadFile(blob, {{ ext: "pdb" }}).then(o => {{
      o.addRepresentation("cartoon", {{ color: "chainname" }});
      o.addRepresentation("surface", {{ opacity: 0.15 }});
      stage.autoView();
      stage.setSpin(true);
      stage.spin(0.6);
    }});
  }});
</script>
"""

# Embed the NGL viewer (requires internet)
html(ngl_html, height=380, scrolling=False)

st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Page: Home (Prediction UI)
# -----------------------------
if page == "Home":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="title">Protein Solubility Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="small">Paste a protein sequence (single-letter amino acids). Click Predict.</div>', unsafe_allow_html=True)
    seq_input = st.text_area("Sequence", height=180, placeholder="Paste sequence here (no spaces/newlines preferred).")
    col1, col2 = st.columns([1,1])
    with col1:
        if st.button("Predict"):
            seq = seq_input.strip().replace("\n", "").replace(" ", "").upper()
            if len(seq) < 10:
                st.error("Please paste a valid protein sequence (at least ~10 aa).")
            else:
                X = build_feature_vector(seq)
                proba = model.predict_proba(X)[0][1]
                pred = "Soluble" if proba >= 0.5 else "Insoluble"
                st.success(f"Prediction: {pred} — Probability (soluble): {proba:.3f}")
                # Small feature snapshot
                fv = pd.Series(X.flatten(), index=FEATURE_COLS)
                top = fv.sort_values(ascending=False).head(6)
                st.markdown("**Feature snapshot:**")
                st.table(top.to_frame("value"))
    with col2:
        st.markdown("### Quick tips")
        st.write("- Short, charged proteins tend to be soluble.")
        st.write("- Membrane proteins and cysteine-rich proteins can be insoluble in E. coli.")
        st.markdown("### Model info")
        st.write("- XGBoost trained on UESolDS-derived features.")
        st.write("- Input features: amino-acid composition + seq length + hydrophobicity + aromatic fraction.")
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Page: Project Details
# -----------------------------
if page == "Project Details":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("Project: Protein Solubility Prediction (UESolDS)")
    st.write("""
    **Goal:** Predict whether a protein expressed in *E. coli* will be soluble or insoluble using sequence-derived features.
    """)
    st.subheader("Dataset")
    st.write("- UESolDS (curated E. coli solubility dataset). Train: ~70k sequences; balanced val/test.")
    st.subheader("Features")
    st.write("- Amino acid composition (20 features).")
    st.write("- Sequence length, hydrophobicity (Kyte-Doolittle avg), aromatic fraction (F/W/Y).")
    st.subheader("Modeling")
    st.write("- Baseline: Logistic Regression (AUC ~0.60).")
    st.write("- Random Forest (AUC ~0.69).")
    st.write("- Final: XGBoost + physchem features (AUC ~0.73).")
    st.subheader("Key findings")
    st.write("- Sequence length and cysteine content are top predictors.")
    st.write("- Charged residues (E, D, R, K) correlate with solubility.")
    st.subheader("How it works (high level)")
    st.write("""
    1. Sequence → compute fixed-length numeric features.
    2. XGBoost predicts probability of being soluble (0-1).
    3. Use prediction as guidance for experimental expression planning.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Page: Samples / Download
# -----------------------------
if page == "Samples / Download":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("Sample Sequences")
    samples = {
        "GFP (soluble)": "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLTYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITLGMDELYK",
        "Thioredoxin (soluble)": "MKKIYFTKGQGPPAVPTTTGRSVPTIEVADKIVVGKPTLLLFKNGEVAATKVGALSKGQLKEFLDANLA",
        "Bacteriorhodopsin (insoluble)": "MNGTEGPNFYVPFSNKTGVVRSPFEYPQYYLAEPWQFSMLAAYMFLLIVLGFPINFLTLYVTVQHKKLRTPLNYILLNLAVADLFMVFGGFTTTLYTSLHGYFVFGPTGCNLEGFFATLGGEIALWSLVVLAFAVYMGVFSLAETNRFGAAHLP",
        "IL-2 (aggregation-prone)": "MYRMQLLSCIALSLALVTNSVTKTEANLAALEAKDSPQTHSLLEDAQQISLDKNQLEHLLLDLQMILNGINNYKNPKLTRMLTFKFYMPKKATELKHLQCLENELGALQR"
    }
    for name, seq in samples.items():
        st.markdown(f"**{name}**")
        st.code(seq, language="text")
        if st.button(f"Use {name}", key=name):
            st.experimental_set_query_params(dummy=name)
            # prefill by writing to the clipboard area (Streamlit doesn't support direct paste)
    st.markdown("### Download model & code")
    st.write("Model saved in `models/xgb2_solubility.joblib` and feature columns in `models/feature_cols.json`.")
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Footer
# -----------------------------
st.markdown('<div class="footer">Built with ❤️ — UESolDS · XGBoost · Streamlit</div>', unsafe_allow_html=True)
