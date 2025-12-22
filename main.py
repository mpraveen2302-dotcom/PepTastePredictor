# ============================================================
# PepTastePredictor – FINAL COMPLETE STREAMLIT APP
# ============================================================

import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import py3Dmol

from collections import Counter
from itertools import product
from Bio.SeqUtils.ProtParam import ProteinAnalysis

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score,
    confusion_matrix, mean_squared_error, r2_score
)
from sklearn.decomposition import PCA

# ============================================================
# APP CONFIG
# ============================================================

st.set_page_config(
    page_title="PepTastePredictor",
    layout="wide",
)

AA = "ACDEFGHIKLMNPQRSTVWY"
DATASET_PATH = "AIML (4).xlsx"

# ============================================================
# SEQUENCE CLEANING
# ============================================================

def clean_sequence(seq):
    if not isinstance(seq, str):
        return ""
    seq = seq.upper().replace("\t", "").replace(" ", "").replace("\n", "")
    return "".join([a for a in seq if a in AA])

# ============================================================
# FEATURE EXTRACTION
# ============================================================

def aa_composition(seq):
    c = Counter(seq)
    L = len(seq)
    return {f"AA_{a}": c.get(a, 0) / L for a in AA}

def dipeptide_composition(seq):
    dipeptides = ["".join(p) for p in product(AA, repeat=2)]
    counts = Counter(seq[i:i+2] for i in range(len(seq)-1))
    L = max(len(seq)-1, 1)
    return {f"DP_{d}": counts.get(d, 0) / L for d in dipeptides}

def physicochemical_features(seq):
    ana = ProteinAnalysis(seq)

    try:
        instab = ana.instability_index()
    except:
        instab = 0.0

    helix, turn, sheet = ana.secondary_structure_fraction()

    return {
        "length": len(seq),
        "molecular_weight": ana.molecular_weight(),
        "isoelectric_point": ana.isoelectric_point(),
        "aromaticity": ana.aromaticity(),
        "instability_index": instab,
        "gravy": ana.gravy(),
        "net_charge_pH7": ana.charge_at_pH(7.0),
        "helix_fraction": helix,
        "turn_fraction": turn,
        "sheet_fraction": sheet,
    }

def extract_features(seqs):
    rows = []
    for s in seqs:
        s = clean_sequence(s)
        if len(s) < 2:
            continue
        f = {}
        f.update(physicochemical_features(s))
        f.update(aa_composition(s))
        f.update(dipeptide_composition(s))
        rows.append(f)
    return pd.DataFrame(rows).fillna(0)

# ============================================================
# PDB BUILDER (PROPER CA TRACE)
# ============================================================

def build_peptide_pdb(seq):
    lines = []
    x = 0.0
    for i, aa in enumerate(seq, start=1):
        lines.append(
            f"ATOM  {i:5d}  CA  {aa} A{i:4d}    "
            f"{x:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00           C"
        )
        x += 3.8
    lines.append("END")
    return "\n".join(lines)

def show_structure(pdb_text):
    view = py3Dmol.view(width=600, height=450)
    view.addModel(pdb_text, "pdb")
    view.setStyle({"cartoon": {"color": "spectrum"}})
    view.zoomTo()
    return view

# ============================================================
# TRAIN MODELS (CACHED)
# ============================================================

@st.cache_data(show_spinner=True)
def train_models():
    df = pd.read_excel(DATASET_PATH)
    df.columns = df.columns.str.lower().str.strip()

    # REQUIRED COLUMN NAMES
    # peptide | taste | solubility | docking score (kcal/mol)

    df["peptide"] = df["peptide"].apply(clean_sequence)
    df = df[df["peptide"].str.len() >= 2].reset_index(drop=True)

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
        "taste_accuracy": accuracy_score(yt_te, taste_model.predict(X_te)),
        "taste_f1": f1_score(yt_te, taste_model.predict(X_te), average="weighted"),
        "sol_accuracy": accuracy_score(ys_te, sol_model.predict(X_te)),
        "sol_f1": f1_score(ys_te, sol_model.predict(X_te), average="weighted"),
        "dock_rmse": np.sqrt(mean_squared_error(yd_te, dock_model.predict(X_te))),
        "dock_r2": r2_score(yd_te, dock_model.predict(X_te)),
    }

    return (
        taste_model, sol_model, dock_model,
        le_taste, le_sol,
        metrics, X_te, yt_te, ys_te, yd_te, X
    )

# ============================================================
# LOAD MODELS
# ============================================================

(
    taste_model, sol_model, dock_model,
    le_taste, le_sol,
    metrics, X_te, yt_te, ys_te, yd_te, X_all
) = train_models()

# ============================================================
# STREAMLIT UI
# ============================================================

st.title("🧬 PepTastePredictor")
st.caption("Comprehensive peptide taste, solubility & docking prediction platform")

# ---------------- SINGLE PREDICTION ----------------

st.header("🔬 Single Peptide Prediction")

seq_input = st.text_input("Enter peptide sequence")

if st.button("Predict"):
    seq = clean_sequence(seq_input)

    if len(seq) < 2:
        st.error("Invalid peptide sequence")
    else:
        Xp = extract_features([seq])
        taste = le_taste.inverse_transform(taste_model.predict(Xp))[0]
        sol = le_sol.inverse_transform(sol_model.predict(Xp))[0]
        dock = dock_model.predict(Xp)[0]

        st.subheader("Prediction")
        st.json({
            "Sequence": seq,
            "Taste": taste,
            "Solubility": sol,
            "Docking Score (kcal/mol)": round(dock, 3),
            "Total Features": X_all.shape[1]
        })

        pdb_text = build_peptide_pdb(seq)
        st.download_button(
            "Download PDB",
            pdb_text,
            file_name="peptide.pdb"
        )

        st.subheader("3D Structure Viewer")
        viewer = show_structure(pdb_text)
        st.components.v1.html(viewer._make_html(), height=450)

# ---------------- BATCH PREDICTION ----------------

st.header("📂 Batch Prediction")

batch_file = st.file_uploader("Upload CSV / Excel with column 'peptide'")

if batch_file:
    dfb = pd.read_excel(batch_file) if batch_file.name.endswith("xlsx") else pd.read_csv(batch_file)
    dfb["peptide"] = dfb["peptide"].apply(clean_sequence)
    Xb = extract_features(dfb["peptide"].tolist())

    dfb["Predicted_Taste"] = le_taste.inverse_transform(taste_model.predict(Xb))
    dfb["Predicted_Solubility"] = le_sol.inverse_transform(sol_model.predict(Xb))
    dfb["Predicted_Docking"] = dock_model.predict(Xb)

    st.dataframe(dfb)
    st.download_button(
        "Download Batch Results",
        dfb.to_csv(index=False),
        file_name="batch_predictions.csv"
    )

# ---------------- ANALYTICS ----------------

st.header("📊 Model Analytics")

st.json(metrics)

col1, col2 = st.columns(2)

with col1:
    fig1, ax1 = plt.subplots()
    sns.heatmap(confusion_matrix(yt_te, taste_model.predict(X_te)),
                annot=True, fmt="d", ax=ax1)
    ax1.set_title("Taste Confusion Matrix")
    st.pyplot(fig1)

with col2:
    fig2, ax2 = plt.subplots()
    sns.heatmap(confusion_matrix(ys_te, sol_model.predict(X_te)),
                annot=True, fmt="d", ax=ax2)
    ax2.set_title("Solubility Confusion Matrix")
    st.pyplot(fig2)

fig3, ax3 = plt.subplots()
coords = PCA(2).fit_transform(X_all)
ax3.scatter(coords[:,0], coords[:,1], c=np.random.rand(len(coords)))
ax3.set_title("PCA Feature Distribution")
st.pyplot(fig3)

fig4, ax4 = plt.subplots()
ax4.scatter(yd_te, dock_model.predict(X_te))
ax4.plot([yd_te.min(), yd_te.max()], [yd_te.min(), yd_te.max()], "r--")
ax4.set_xlabel("Actual Docking")
ax4.set_ylabel("Predicted Docking")
ax4.set_title("Docking Regression")
st.pyplot(fig4)

# ---------------- ALPHAFOLD ----------------

st.header("🧬 AlphaFold / ColabFold")
st.markdown(
    "Generate high-quality structures:\n\n"
    "🔗 https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/AlphaFold2_mmseqs2_advanced.ipynb"
)
