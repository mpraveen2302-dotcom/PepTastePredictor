# ==========================================================
# PepTastePredictor — FINAL PUBLICATION VERSION (STABLE + MODE SELECT)
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
# FRONTEND STYLING (UI ONLY)
# ==========================================================

st.markdown("""
<style>
.stApp { background-color: #f4f7fb; }
h1, h2, h3 { color: #1f3c88; }

.hero {
    background: linear-gradient(90deg, #1f3c88, #0b7285);
    padding: 30px;
    border-radius: 16px;
    color: white;
    margin-bottom: 30px;
}

.badge {
    display: inline-block;
    background-color: rgba(255,255,255,0.15);
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 13px;
    margin-right: 8px;
}

.card {
    background: white;
    padding: 22px;
    border-radius: 14px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.08);
    margin-bottom: 24px;
}

.metric {
    font-size: 20px;
    font-weight: 600;
    color: #0b7285;
}

.footer {
    text-align: center;
    color: #6c757d;
    font-size: 13px;
    padding: 30px;
}
</style>
""", unsafe_allow_html=True)

# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.image("logo.png", width=120)
st.sidebar.markdown("### PepTastePredictor")
st.sidebar.write("AI-powered peptide analysis platform")
st.sidebar.markdown("---")
st.sidebar.info("For research & educational use")

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
# FEATURE EXTRACTION (RESTORED)
# ==========================================================

def model_features(seq):
    ana = ProteinAnalysis(seq)
    f = {
        "length": len(seq),
        "mw": ana.molecular_weight(),
        "pI": ana.isoelectric_point(),
        "aromaticity": ana.aromaticity(),
        "instability": ana.instability_index(),
        "gravy": ana.gravy(),
        "charge": ana.charge_at_pH(7.0)
    }
    for aa in AA:
        f[f"AA_{aa}"] = seq.count(aa) / len(seq)
    return f

def build_feature_table(seqs):
    return pd.DataFrame([model_features(s) for s in seqs]).fillna(0)

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
# TRAIN MODELS (UNCHANGED)
# ==========================================================

@st.cache_data
def train_models():
    df = pd.read_excel(DATASET_PATH)
    df.columns = df.columns.str.lower().str.strip()
    df["peptide"] = df["peptide"].apply(clean_sequence)
    df = df[df["peptide"].str.len() >= 2]

    X = build_feature_table(df["peptide"])
    le_taste, le_sol = LabelEncoder(), LabelEncoder()

    y_taste = le_taste.fit_transform(df["taste"])
    y_sol = le_sol.fit_transform(df["solubility"])
    y_dock = df["docking score (kcal/mol)"]

    Xtr, Xte, yt_tr, yt_te, ys_tr, ys_te, yd_tr, yd_te = train_test_split(
        X, y_taste, y_sol, y_dock, test_size=0.2, random_state=42
    )

    taste_model = RandomForestClassifier(n_estimators=300)
    sol_model = RandomForestClassifier(n_estimators=300)
    dock_model = RandomForestRegressor(n_estimators=400)

    taste_model.fit(Xtr, yt_tr)
    sol_model.fit(Xtr, ys_tr)
    dock_model.fit(Xtr, yd_tr)

    return df, X, taste_model, sol_model, dock_model, le_taste, le_sol

df, X_all, taste_model, sol_model, dock_model, le_taste, le_sol = train_models()

# ==========================================================
# HERO HEADER
# ==========================================================

st.markdown("""
<div class="hero">
    <h1>🧬 PepTastePredictor</h1>
    <p>AI-powered peptide taste, solubility & docking prediction platform</p>
    <span class="badge">Machine Learning</span>
    <span class="badge">Peptide Science</span>
    <span class="badge">Structural Biology</span>
</div>
""", unsafe_allow_html=True)

# ==========================================================
# MODE SELECTION (AS YOU ASKED)
# ==========================================================

mode = st.radio(
    "Select prediction mode",
    ["Single Peptide Prediction", "Batch Peptide Prediction"],
    horizontal=True
)

# ==========================================================
# SINGLE PEPTIDE PREDICTION
# ==========================================================

if mode == "Single Peptide Prediction":

    st.markdown("## 🔬 Single Peptide Prediction")

    seq = st.text_input("Enter peptide sequence")

    if st.button("Predict"):
        seq = clean_sequence(seq)
        Xp = pd.DataFrame([model_features(seq)])

        taste = le_taste.inverse_transform(taste_model.predict(Xp))[0]
        sol = le_sol.inverse_transform(sol_model.predict(Xp))[0]
        dock = dock_model.predict(Xp)[0]

        st.markdown(f"""
        <div class="card">
            <div class="metric">Taste: {taste}</div>
            <div class="metric">Solubility: {sol}</div>
            <div class="metric">Docking score: {dock:.2f} kcal/mol</div>
        </div>
        """, unsafe_allow_html=True)

        st.write("### Physicochemical Properties")
        for k, v in physicochemical_features(seq).items():
            st.write(f"{k}: {v}")

        st.write("### Composition Summary")
        for k, v in composition_features(seq).items():
            st.write(f"{k}: {v}")

# ==========================================================
# BATCH PEPTIDE PREDICTION
# ==========================================================

if mode == "Batch Peptide Prediction":

    st.markdown("## 📦 Batch Peptide Prediction")

    batch_file = st.file_uploader("Upload CSV with 'peptide' column", type=["csv"])

    if batch_file:
        batch_df = pd.read_csv(batch_file)

        if "peptide" not in batch_df.columns:
            st.error("CSV must contain a column named 'peptide'")
        else:
            batch_df["peptide"] = batch_df["peptide"].apply(clean_sequence)
            Xb = build_feature_table(batch_df["peptide"])

            batch_df["Predicted Taste"] = le_taste.inverse_transform(taste_model.predict(Xb))
            batch_df["Predicted Solubility"] = le_sol.inverse_transform(sol_model.predict(Xb))
            batch_df["Predicted Docking Score"] = dock_model.predict(Xb)

            st.dataframe(batch_df)
            st.download_button(
                "Download Batch Predictions",
                batch_df.to_csv(index=False),
                "batch_predictions.csv"
            )

# ==========================================================
# FOOTER
# ==========================================================

st.markdown("""
<div class="footer">
© 2025 <b>PepTastePredictor</b><br>
AI-driven peptide analysis platform for research, education & innovation
</div>
""", unsafe_allow_html=True)
