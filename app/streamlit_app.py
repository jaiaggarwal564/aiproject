# app/streamlit_app.py
import streamlit as st
import joblib
import json
import pandas as pd
import numpy as np

st.set_page_config(page_title="Protein Solubility Predictor", layout="centered")

st.title("Protein Solubility Predictor — UESolDS / XGBoost")
st.write("Paste a protein sequence (single-letter amino acids). The model predicts Soluble vs Insoluble.")

# Load model + feature cols
MODEL_PATH = "D:/protein_solubility_project/models/xgb2_solubility.joblib"
FEATURE_COLS_PATH = "D:/protein_solubility_project/models/feature_cols.json"

@st.cache_resource
def load_model_and_cols():
    model = joblib.load(MODEL_PATH)
    with open(FEATURE_COLS_PATH, "r") as f:
        cols = json.load(f)
    return model, cols

model, FEATURE_COLS = load_model_and_cols()

# --- helper functions (same as notebook) ---
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
    # Combine in the same order as FEATURE_COLS
    row = []
    for c in FEATURE_COLS:
        if c in aa_feats:
            row.append(aa_feats[c])
        elif c in phys:
            row.append(phys[c])
        else:
            # unknown column -> zero
            row.append(0.0)
    return np.array(row).reshape(1, -1)

# --- UI ---
st.markdown("### Input sequence")
seq_input = st.text_area("Paste sequence here (single FASTA sequence, no spaces/newlines preferred).", height=150)

col1, col2 = st.columns([1,1])
with col1:
    if st.button("Predict"):
        seq = seq_input.strip().replace("\n", "").replace(" ", "")
        if len(seq) < 10:
            st.error("Please paste a valid protein sequence (at least ~10 aa).")
        else:
            X = build_feature_vector(seq)
            proba = model.predict_proba(X)[0][1]  # probability soluble
            pred = "Soluble" if proba >= 0.5 else "Insoluble"
            st.subheader(f"Prediction: {pred}")
            st.write(f"Soluble probability: **{proba:.3f}**")
            # Show top contributions (approx) by comparing to mean feature vector
            try:
                # rough feature effect: show top 6 features where input differs most from training mean
                # Load training mean from model if saved; otherwise estimate from FEATURE_COLS (fallback zeros)
                st.markdown("#### Feature snapshot (selected):")
                fv = pd.Series(X.flatten(), index=FEATURE_COLS)
                key_feats = fv.sort_values(ascending=False).head(6)
                st.table(key_feats.to_frame("value"))
            except Exception:
                pass

with col2:
    st.markdown("### Example sequences")
    st.write("- Try a small soluble enzyme sequence or a long membrane protein to see differences.")
    st.markdown("### Model info")
    st.write("- Model: XGBoost trained on UESolDS-derived features.")
    st.write("- Features: amino-acid composition + seq length + hydrophobicity + aromatic fraction.")

st.markdown("---")
st.markdown("#### Notes / Caveats")
st.write("""
- This model predicts solubility *as observed in E. coli expression experiments* used to build UESolDS.
- Predictions are probabilistic and should be used as guidance, not definitive truth.
""")
