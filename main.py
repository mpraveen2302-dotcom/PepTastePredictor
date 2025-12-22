# ============================================================
# PepTastePredictor – FINAL STABLE STREAMLIT APP
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import py3Dmol
import io

from collections import Counter
from itertools import product

from Bio.SeqUtils.ProtParam import ProteinAnalysis

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, r2_score, mean_squared_error
from sklearn.decomposition import PCA

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
st.set_page_config(page_title="PepTastePredictor", layout="wide")
DATASET_PATH = "AIML (4).xlsx"
AA = "ACDEFGHIKLMNPQRSTVWY"

# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------
def clean_sequence(seq):
    seq = str(seq).upper().replace(" ", "").replace("\t", "").replace("\n", "")
    return "".join(a for a in seq if a in AA)

def build_ca_pdb(seq):
    lines, x = [], 0.0
    for i, aa in enumerate(seq, 1):
        lines.append(
            f"ATOM  {i:5d}  CA  {aa} A{i:4d}    {x:8.3f} 0.000 0.000  1.00  0.00           C"
        )
        x += 3.8
    return "\n".join(lines) + "\nEND\n"

def show_structure(pdb_text):
    view = py3Dmol.view(width=700, height=500)
    view.addModel(pdb_text, "pdb")
    view.setStyle({"cartoon": {"color": "spectrum"}})
    view.zoomTo()
    return view

# ------------------------------------------------------------
# FEATURE EXTRACTION
# ------------------------------------------------------------
def extract_features(seq):
    ana = ProteinAnalysis(seq)
    helix, turn, sheet = ana.secondary_structure_fraction()

    feats = {
        "Length": len(seq),
        "Molecular Weight": ana.molecular_weight(),
        "Isoelectric Point": ana.isoelectric_point(),
        "Aromaticity": ana.aromaticity(),
        "Instability Index": ana.instability_index(),
        "GRAVY": ana.gravy(),
        "Net Charge (pH 7)": ana.charge_at_pH(7.0),
        "Helix Fraction": helix,
        "Turn Fraction": turn,
        "Sheet Fraction": sheet
    }

    # Amino acid composition
    counts = Counter(seq)
    for aa in AA:
        feats[f"AA_{aa}"] = counts.get(aa, 0) / len(seq)

    return feats

def features_dataframe(seqs):
    return pd.DataFrame([extract_features(s) for s in seqs]).fillna(0)

# ------------------------------------------------------------
# LOAD & TRAIN
# ------------------------------------------------------------
@st.cache_resource
def train_models():
    df = pd.read_excel(DATASET_PATH)
    df.columns = df.columns.str.lower().str.strip()
    df["peptide"] = df["peptide"].apply(clean_sequence)
    df = df[df["peptide"].str.len() >= 2]

    X = features_dataframe(df["peptide"])
    le_taste, le_sol = LabelEncoder(), LabelEncoder()

    y_taste = le_taste.fit_transform(df["taste"])
    y_sol = le_sol.fit_transform(df["solubility"])
    y_dock = df["docking score (kcal/mol)"].astype(float)

    Xtr, Xte, yt_tr, yt_te, ys_tr, ys_te, yd_tr, yd_te = train_test_split(
        X, y_taste, y_sol, y_dock, test_size=0.2, random_state=42
    )

    taste_model = RandomForestClassifier(n_estimators=300)
    sol_model = RandomForestClassifier(n_estimators=300)
    dock_model = RandomForestRegressor(n_estimators=400)

    taste_model.fit(Xtr, yt_tr)
    sol_model.fit(Xtr, ys_tr)
    dock_model.fit(Xtr, yd_tr)

    metrics = {
        "Taste Accuracy": accuracy_score(yt_te, taste_model.predict(Xte)),
        "Taste F1": f1_score(yt_te, taste_model.predict(Xte), average="weighted"),
        "Solubility Accuracy": accuracy_score(ys_te, sol_model.predict(Xte)),
        "Solubility F1": f1_score(ys_te, sol_model.predict(Xte), average="weighted"),
        "Docking RMSE": np.sqrt(mean_squared_error(yd_te, dock_model.predict(Xte))),
        "Docking R2": r2_score(yd_te, dock_model.predict(Xte))
    }

    return taste_model, sol_model, dock_model, le_taste, le_sol, metrics, Xte, yt_te, ys_te, yd_te

taste_model, sol_model, dock_model, le_taste, le_sol, metrics, Xte, yt_te, ys_te, yd_te = train_models()

# ------------------------------------------------------------
# UI
# ------------------------------------------------------------
st.title("🧬 PepTastePredictor")

# ================= SINGLE PREDICTION =================
st.header("🔹 Single Peptide Prediction")
seq = st.text_input("Enter peptide sequence")

if st.button("Predict"):
    seq = clean_sequence(seq)
    feats = extract_features(seq)
    Xp = pd.DataFrame([feats])

    taste = le_taste.inverse_transform(taste_model.predict(Xp))[0]
    sol = le_sol.inverse_transform(sol_model.predict(Xp))[0]
    dock = dock_model.predict(Xp)[0]

    st.subheader("Prediction Summary")
    st.write(f"**Taste:** {taste}")
    st.write(f"**Solubility:** {sol}")
    st.write(f"**Docking Score:** {dock:.2f} kcal/mol")

    st.subheader("Physiochemical Properties")
    for k, v in feats.items():
        if not k.startswith("AA_"):
            st.write(f"• **{k}:** {v}")

    pdb_text = build_ca_pdb(seq)
    st.download_button("Download Peptide PDB", pdb_text, "peptide.pdb")

    st.subheader("3D Structure Viewer")
    viewer = show_structure(pdb_text)
    st.components.v1.html(viewer._make_html(), height=500)

# ================= BATCH =================
st.header("📂 Batch Prediction")
batch = st.file_uploader("Upload CSV (column: peptide)", type=["csv"])

if batch:
    dfb = pd.read_csv(batch)
    dfb["peptide"] = dfb["peptide"].apply(clean_sequence)
    Xb = features_dataframe(dfb["peptide"])

    dfb["Taste"] = le_taste.inverse_transform(taste_model.predict(Xb))
    dfb["Solubility"] = le_sol.inverse_transform(sol_model.predict(Xb))
    dfb["Docking Score"] = dock_model.predict(Xb)

    st.dataframe(dfb)
    st.download_button("Download Batch Results", dfb.to_csv(index=False), "batch_results.csv")

# ================= ANALYTICS =================
st.header("📊 Model Analytics")

fig1, ax1 = plt.subplots()
sns.heatmap(confusion_matrix(yt_te, taste_model.predict(Xte)), annot=True, ax=ax1)
ax1.set_title("Taste Confusion Matrix")
st.pyplot(fig1)

fig2, ax2 = plt.subplots()
sns.heatmap(confusion_matrix(ys_te, sol_model.predict(Xte)), annot=True, ax=ax2)
ax2.set_title("Solubility Confusion Matrix")
st.pyplot(fig2)

fig3, ax3 = plt.subplots()
ax3.scatter(yd_te, dock_model.predict(Xte))
ax3.plot([yd_te.min(), yd_te.max()], [yd_te.min(), yd_te.max()], "r--")
ax3.set_title("Docking Regression")
st.pyplot(fig3)

# ================= ALPHAFOLD =================
st.header("🧬 AlphaFold / ColabFold Structure")

st.markdown(
    "Use ColabFold to generate predicted structures:\n"
    "https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/AlphaFold2_mmseqs2_advanced.ipynb"
)

uploaded_pdb = st.file_uploader("Upload AlphaFold/ColabFold PDB", type=["pdb"])

if uploaded_pdb:
    pdb_txt = uploaded_pdb.read().decode("utf-8")
    viewer = show_structure(pdb_txt)
    st.components.v1.html(viewer._make_html(), height=500)
else:
    st.info("No PDB uploaded yet.")
