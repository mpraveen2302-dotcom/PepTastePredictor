# ============================================================
# PepTastePredictor – FINAL USER-FRIENDLY RESEARCH APP
# ============================================================

import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import py3Dmol
from io import StringIO

from collections import Counter
from itertools import product
from Bio.SeqUtils.ProtParam import ProteinAnalysis

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, mean_squared_error, r2_score
from sklearn.decomposition import PCA

# ============================================================
# CONFIG
# ============================================================

DATASET_PATH = "AIML (4).xlsx"
AA = "ACDEFGHIKLMNPQRSTVWY"

st.set_page_config(page_title="PepTastePredictor", layout="wide")

# ============================================================
# SEQUENCE CLEANING
# ============================================================

def clean_sequence(seq):
    if not isinstance(seq, str):
        return ""
    seq = seq.upper().replace(" ", "").replace("\n", "").replace("\t", "")
    return "".join([a for a in seq if a in AA])

# ============================================================
# FEATURE ENGINEERING (400+ FEATURES)
# ============================================================

def aa_composition(seq):
    c = Counter(seq)
    L = len(seq)
    return {f"AA_{a}": c.get(a, 0)/L for a in AA}

def dipeptide_composition(seq):
    dipeptides = ["".join(p) for p in product(AA, repeat=2)]
    counts = Counter(seq[i:i+2] for i in range(len(seq)-1))
    L = max(len(seq)-1, 1)
    return {f"DP_{d}": counts.get(d, 0)/L for d in dipeptides}

def biophysical_features(seq):
    ana = ProteinAnalysis(seq)
    return {
        "length": len(seq),
        "molecular_weight": ana.molecular_weight(),
        "isoelectric_point": ana.isoelectric_point(),
        "aromaticity": ana.aromaticity(),
        "instability": ana.instability_index() if len(seq) > 1 else 0,
        "gravy": ana.gravy(),
        "charge_pH7": ana.charge_at_pH(7.0),
        "helix": ana.secondary_structure_fraction()[0],
        "turn": ana.secondary_structure_fraction()[1],
        "sheet": ana.secondary_structure_fraction()[2],
    }

def extract_features(seqs):
    feats = []
    for s in seqs:
        f = {}
        f.update(biophysical_features(s))
        f.update(aa_composition(s))
        f.update(dipeptide_composition(s))
        feats.append(f)
    return pd.DataFrame(feats).fillna(0)

# ============================================================
# LOAD DATA & TRAIN MODELS
# ============================================================

@st.cache_resource
def train_models():
    df = pd.read_excel(DATASET_PATH)
    df.columns = df.columns.str.lower().str.strip()
    df["peptide"] = df["peptide"].apply(clean_sequence)
    df = df[df["peptide"].str.len() >= 2]

    X = extract_features(df["peptide"].tolist())

    le_taste = LabelEncoder()
    le_sol = LabelEncoder()

    y_taste = le_taste.fit_transform(df["taste"])
    y_sol = le_sol.fit_transform(df["solubility"])
    y_dock = df["docking score (kcal/mol)"].astype(float)

    strat = y_taste if min(Counter(y_taste).values()) >= 2 else None

    X_tr, X_te, yt_tr, yt_te, ys_tr, ys_te, yd_tr, yd_te = train_test_split(
        X, y_taste, y_sol, y_dock,
        test_size=0.2,
        random_state=42,
        stratify=strat
    )

    taste_model = RandomForestClassifier(n_estimators=300, random_state=42)
    sol_model = RandomForestClassifier(n_estimators=300, random_state=42)
    dock_model = RandomForestRegressor(n_estimators=400, random_state=42)

    taste_model.fit(X_tr, yt_tr)
    sol_model.fit(X_tr, ys_tr)
    dock_model.fit(X_tr, yd_tr)

    metrics = {
        "Taste Accuracy": accuracy_score(yt_te, taste_model.predict(X_te)),
        "Taste F1": f1_score(yt_te, taste_model.predict(X_te), average="weighted"),
        "Solubility Accuracy": accuracy_score(ys_te, sol_model.predict(X_te)),
        "Solubility F1": f1_score(ys_te, sol_model.predict(X_te), average="weighted"),
        "Docking RMSE": np.sqrt(mean_squared_error(yd_te, dock_model.predict(X_te))),
        "Docking R2": r2_score(yd_te, dock_model.predict(X_te)),
    }

    return df, X, X_te, yt_te, ys_te, yd_te, taste_model, sol_model, dock_model, le_taste, le_sol, metrics

(df, X_all, X_te, yt_te, ys_te, yd_te,
 taste_model, sol_model, dock_model,
 le_taste, le_sol, metrics) = train_models()

TOTAL_FEATURES = X_all.shape[1]

# ============================================================
# PDB GENERATION & VIEWER
# ============================================================

def build_pdb(seq):
    lines = []
    x = 0.0
    for i, aa in enumerate(seq, 1):
        lines.append(
            f"ATOM  {i:5d}  CA  {aa} A{i:4d}    {x:8.3f} 0.000 0.000  1.00  0.00           C"
        )
        x += 3.8
    return "\n".join(lines) + "\nEND"

def show_structure(pdb_text):
    view = py3Dmol.view(width=600, height=450)
    view.addModel(pdb_text, "pdb")
    view.setStyle({"cartoon": {"color": "spectrum"}})
    view.zoomTo()
    return view

# ============================================================
# UI
# ============================================================

st.title("🧬 PepTastePredictor")
st.markdown(f"**Total Features Used:** `{TOTAL_FEATURES}`")

# ------------------------------------------------------------
# 1️⃣ SINGLE PEPTIDE PREDICTION
# ------------------------------------------------------------

st.header("🔬 Single Peptide Prediction")

seq = st.text_input("Enter Peptide Sequence")

if st.button("Predict Peptide") and seq:
    seq = clean_sequence(seq)
    Xp = extract_features([seq])

    taste = le_taste.inverse_transform(taste_model.predict(Xp))[0]
    sol = le_sol.inverse_transform(sol_model.predict(Xp))[0]
    dock = dock_model.predict(Xp)[0]

    pdb_text = build_pdb(seq)
    st.json({
        "Sequence": seq,
        "Taste": taste,
        "Solubility": sol,
        "Docking Score (kcal/mol)": round(dock, 3)
    })

    st.download_button("⬇ Download PDB", pdb_text, "peptide.pdb")

    viewer = show_structure(pdb_text)
    st.components.v1.html(viewer._make_html(), height=480)

# ------------------------------------------------------------
# 2️⃣ BATCH PREDICTION
# ------------------------------------------------------------

st.header("📂 Batch Prediction")

batch_file = st.file_uploader("Upload CSV / Excel (must contain 'peptide' column)",
                              type=["csv", "xlsx"])

if batch_file:
    if batch_file.name.endswith(".csv"):
        batch_df = pd.read_csv(batch_file)
    else:
        batch_df = pd.read_excel(batch_file)

    batch_df["peptide"] = batch_df["peptide"].apply(clean_sequence)
    Xb = extract_features(batch_df["peptide"].tolist())

    batch_df["Predicted Taste"] = le_taste.inverse_transform(taste_model.predict(Xb))
    batch_df["Predicted Solubility"] = le_sol.inverse_transform(sol_model.predict(Xb))
    batch_df["Predicted Docking Score"] = dock_model.predict(Xb)

    st.dataframe(batch_df)

    csv = batch_df.to_csv(index=False)
    st.download_button("⬇ Download Batch Results", csv, "batch_predictions.csv")

# ------------------------------------------------------------
# 3️⃣ MODEL ANALYSIS & VISUALIZATIONS
# ------------------------------------------------------------

st.header("📊 Model Analysis & Visualizations")

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(yt_te, taste_model.predict(X_te)),
                annot=True, fmt="d", ax=ax,
                xticklabels=le_taste.classes_,
                yticklabels=le_taste.classes_)
    ax.set_title("Taste Confusion Matrix")
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(ys_te, sol_model.predict(X_te)),
                annot=True, fmt="d", ax=ax,
                xticklabels=le_sol.classes_,
                yticklabels=le_sol.classes_)
    ax.set_title("Solubility Confusion Matrix")
    st.pyplot(fig)

fig, ax = plt.subplots()
coords = PCA(2).fit_transform(X_all)
ax.scatter(coords[:,0], coords[:,1], c=df["taste"].factorize()[0], cmap="tab10")
ax.set_title("PCA Feature Space")
st.pyplot(fig)

fig, ax = plt.subplots()
ax.scatter(yd_te, dock_model.predict(X_te))
ax.plot([yd_te.min(), yd_te.max()], [yd_te.min(), yd_te.max()], "r--")
ax.set_title("Docking: Actual vs Predicted")
st.pyplot(fig)

# ------------------------------------------------------------
# 4️⃣ DOWNLOADABLE ANALYTICS REPORT
# ------------------------------------------------------------

st.header("📄 Download Analytics Report")

report = StringIO()
report.write("PepTastePredictor – Analytics Report\n\n")
report.write(f"Total Features: {TOTAL_FEATURES}\n")
report.write(f"Dataset Size: {len(df)}\n\n")
for k, v in metrics.items():
    report.write(f"{k}: {v}\n")

st.download_button("⬇ Download Report", report.getvalue(), "analytics_report.txt")

st.markdown(
    "**AlphaFold / ColabFold Notebook:**  \n"
    "https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/AlphaFold2_mmseqs2_advanced.ipynb"
)
