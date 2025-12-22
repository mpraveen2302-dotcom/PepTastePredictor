# ==========================================================
# PepTastePredictor — FINAL PUBLICATION VERSION (PREMIUM UI)
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import py3Dmol

from collections import Counter
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio.PDB import PDBIO, PDBParser, PPBuilder
import PeptideBuilder
from PeptideBuilder import Geometry

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.decomposition import PCA

# ==========================================================
# CONFIG
# ==========================================================

st.set_page_config(
    page_title="PepTastePredictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

DATASET_PATH = "AIML (4).xlsx"
AA = "ACDEFGHIKLMNPQRSTVWY"

# ==========================================================
# PREMIUM FRONTEND STYLING (NO BACKEND IMPACT)
# ==========================================================

st.markdown("""
<style>

/* App background */
.stApp {
    background-color: #f4f7fb;
}

/* Hero section */
.hero {
    background: linear-gradient(90deg, #1f3c88, #0b7285);
    padding: 30px;
    border-radius: 16px;
    color: white;
    margin-bottom: 30px;
}

/* Badges */
.badge {
    display: inline-block;
    background-color: rgba(255,255,255,0.15);
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 13px;
    margin-right: 8px;
}

/* Cards */
.card {
    background: white;
    padding: 22px;
    border-radius: 14px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.08);
    margin-bottom: 24px;
}

/* Section headers */
.section-title {
    color: #1f3c88;
    font-size: 24px;
    font-weight: 700;
    margin-bottom: 12px;
}

/* Metrics */
.metric {
    font-size: 20px;
    font-weight: 600;
    color: #0b7285;
    margin-bottom: 6px;
}

/* Footer */
.footer {
    text-align: center;
    color: #6c757d;
    font-size: 13px;
    padding: 30px;
}

/* Buttons */
div.stButton > button {
    background-color: #1f3c88;
    color: white;
    border-radius: 8px;
    padding: 0.6em 1.6em;
    border: none;
}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# SIDEBAR (OFFICIAL-STYLE INFO PANEL)
# ==========================================================

st.sidebar.image("logo.png", width=120)
st.sidebar.markdown("### PepTastePredictor")
st.sidebar.markdown("""
**AI-driven peptide analysis platform**

• Taste prediction  
• Solubility prediction  
• Docking score estimation  
• 3D structure visualization  
• Batch peptide screening  

**Intended for**
- Academic research
- iGEM & outreach
- Drug discovery
- Educational demonstrations
""")
st.sidebar.markdown("---")
st.sidebar.info("For research & educational use only")

# ==========================================================
# SESSION STATE
# ==========================================================

if "pdb_text" not in st.session_state:
    st.session_state.pdb_text = None

# ==========================================================
# UTILITIES
# ==========================================================

def clean_sequence(seq):
    if not isinstance(seq, str):
        return ""
    seq = seq.upper().replace(" ", "").replace("\n", "").replace("\t", "")
    return "".join(a for a in seq if a in AA)

# ==========================================================
# FEATURE EXTRACTION
# ==========================================================

def physicochemical_features(seq):
    ana = ProteinAnalysis(seq)
    helix, turn, sheet = ana.secondary_structure_fraction()
    return {
        "Length": len(seq),
        "Molecular weight (Da)": round(ana.molecular_weight(), 2),
        "Isoelectric point (pI)": round(ana.isoelectric_point(), 2),
        "Net charge (pH 7)": round(ana.charge_at_pH(7.0), 2),
        "Aromaticity": round(ana.aromaticity(), 3),
        "GRAVY": round(ana.gravy(), 3),
        "Instability index": round(ana.instability_index(), 2),
        "Helix fraction": round(helix, 3),
        "Turn fraction": round(turn, 3),
        "Sheet fraction": round(sheet, 3),
    }

def composition_features(seq):
    c = Counter(seq)
    L = len(seq)
    return {
        "Hydrophobic (%)": round(100 * sum(c[a] for a in "AILMFWV") / L, 1),
        "Polar (%)": round(100 * sum(c[a] for a in "STNQ") / L, 1),
        "Charged (%)": round(100 * sum(c[a] for a in "DEKRH") / L, 1),
        "Aromatic (%)": round(100 * sum(c[a] for a in "FWY") / L, 1),
    }

# ==========================================================
# HERO HEADER (LOGO + TAGLINE)
# ==========================================================

st.markdown("""
<div class="hero">
    <h1>🧬 PepTastePredictor</h1>
    <p>
    AI-powered peptide taste, solubility & docking prediction platform
    integrating machine learning with structural bioinformatics
    </p>
    <div>
        <span class="badge">Machine Learning</span>
        <span class="badge">Peptide Science</span>
        <span class="badge">Structural Biology</span>
        <span class="badge">Bioinformatics</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ==========================================================
# SINGLE PREDICTION (UI CARD)
# ==========================================================

st.markdown("<div class='section-title'>🔬 Single Peptide Prediction</div>", unsafe_allow_html=True)

seq = st.text_input("Enter peptide sequence (e.g., AGLWFK)", key="single_seq")

if st.button("Predict", key="predict_btn"):

    seq = clean_sequence(seq)
    Xp = pd.DataFrame([model_features(seq)])

    taste = le_taste.inverse_transform(taste_model.predict(Xp))[0]
    sol = le_sol.inverse_transform(sol_model.predict(Xp))[0]
    dock = dock_model.predict(Xp)[0]

    st.markdown(f"""
    <div class="card">
        <div class="metric">Predicted Taste: {taste}</div>
        <div class="metric">Predicted Solubility: {sol}</div>
        <div class="metric">Docking Score: {dock:.2f} kcal/mol</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Physicochemical Properties**")
    for k, v in physicochemical_features(seq).items():
        st.write(f"{k}: {v}")

    st.markdown("**Composition Summary**")
    for k, v in composition_features(seq).items():
        st.write(f"{k}: {v}")

# ==========================================================
# FOOTER (OFFICIAL LOOK)
# ==========================================================

st.markdown("""
<div class="footer">
© 2025 <b>PepTastePredictor</b><br>
AI-driven peptide analysis platform for research, education & innovation
</div>
""", unsafe_allow_html=True)
