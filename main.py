# ==========================================
# PepTastePredictor – FINAL COMPLETE APP
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
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

# ------------------------------------------
# PAGE CONFIG
# ------------------------------------------
st.set_page_config(
    page_title="PepTastePredictor",
    layout="wide"
)

st.title("🧬 PepTastePredictor")
st.markdown(
    """
    **Unified AI platform for peptide taste prediction, solubility analysis,  
    docking score estimation, and 3D structure visualization**
    """
)

# ------------------------------------------
# CONSTANTS
# ------------------------------------------
DATASET_PATH = "AIML (4).xlsx"
AA = "ACDEFGHIKLMNPQRSTVWY"

# ------------------------------------------
# SEQUENCE CLEANING
# ------------------------------------------
def clean_sequence(seq):
    if not isinstance(seq, str):
        return ""
    seq = seq.upper().replace(" ", "").replace("\n", "").replace("\t", "")
    return "".join([a for a in seq if a in AA])

# ------------------------------------------
# FEATURE ENGINEERING
# ------------------------------------------
def aa_composition(seq):
    c = Counter(seq)
    L = len(seq)
    return {f"AA_{a}": c.get(a, 0)/L for a in AA}

def dipeptide_composition(seq):
    dipeptides = ["".join(p) for p in product(AA, repeat=2)]
    counts = Counter(seq[i:i+2] for i in range(len(seq)-1))
    L = max(len(seq)-1, 1)
    return {f"DP_{d}": counts.get(d, 0)/L for d in dipeptides}

def biopython_features(seq):
    ana = ProteinAnalysis(seq)
    helix, turn, sheet = ana.secondary_structure_fraction()
    try:
        instab = ana.instability_index()
    except:
        instab = 0.0

    return {
        "Length": len(seq),
        "Molecular Weight": ana.molecular_weight(),
        "Isoelectric Point": ana.isoelectric_point(),
        "Aromaticity": ana.aromaticity(),
        "Instability Index": instab,
        "GRAVY": ana.gravy(),
        "Net Charge (pH 7)": ana.charge_at_pH(7.0),
        "Helix Fraction": helix,
        "Turn Fraction": turn,
        "Sheet Fraction": sheet,
    }

def extract_features(seqs):
    rows = []
    for s in seqs:
        s = clean_sequence(s)
        f = {}
        f.update(biopython_features(s))
        f.update(aa_composition(s))
        f.update(dipeptide_composition(s))
        rows.append(f)
    return pd.DataFrame(rows).fillna(0)

# ------------------------------------------
# LOAD DATA
# ------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_excel(DATASET_PATH)
    df.columns = df.columns.str.lower().str.strip()
    df["peptide"] = df["peptide"].apply(clean_sequence)
    df = df[df["peptide"].str.len() >= 2]
    return df

df = load_data()

# ------------------------------------------
# TRAIN MODELS
# ------------------------------------------
@st.cache_resource
def train_models(df):
    X = extract_features(df["peptide"].tolist())

    le_taste = LabelEncoder()
    le_sol = LabelEncoder()

    y_taste = le_taste.fit_transform(df["taste"])
    y_sol = le_sol.fit_transform(df["solubility"])
    y_dock = df["docking score (kcal/mol)"].astype(float)

    strat = y_taste if min(pd.Series(y_taste).value_counts()) >= 2 else None

    Xtr, Xte, yt_tr, yt_te, ys_tr, ys_te, yd_tr, yd_te = train_test_split(
        X, y_taste, y_sol, y_dock,
        test_size=0.2,
        random_state=42,
        stratify=strat
    )

    taste_model = RandomForestClassifier(n_estimators=300, random_state=42)
    sol_model = RandomForestClassifier(n_estimators=300, random_state=42)
    dock_model = RandomForestRegressor(n_estimators=400, random_state=42)

    taste_model.fit(Xtr, yt_tr)
    sol_model.fit(Xtr, ys_tr)
    dock_model.fit(Xtr, yd_tr)

    metrics = {
        "Taste Accuracy": accuracy_score(yt_te, taste_model.predict(Xte)),
        "Taste F1": f1_score(yt_te, taste_model.predict(Xte), average="weighted"),
        "Solubility Accuracy": accuracy_score(ys_te, sol_model.predict(Xte)),
        "Solubility F1": f1_score(ys_te, sol_model.predict(Xte), average="weighted"),
        "Docking RMSE": np.sqrt(mean_squared_error(yd_te, dock_model.predict(Xte))),
        "Docking R²": r2_score(yd_te, dock_model.predict(Xte)),
    }

    return X, Xte, yt_te, ys_te, yd_te, taste_model, sol_model, dock_model, le_taste, le_sol, metrics

(X_all, X_test, yt_te, ys_te, yd_te,
 taste_model, sol_model, dock_model,
 le_taste, le_sol, metrics) = train_models(df)

# ------------------------------------------
# 3D STRUCTURE BUILDER
# ------------------------------------------
def build_peptide_pdb(seq):
    lines = []
    x = 0.0
    for i, aa in enumerate(seq, 1):
        lines.append(
            f"ATOM  {i:5d}  CA  {aa} A{i:4d}    {x:8.3f} 0.000 0.000  1.00  0.00           C"
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

# ------------------------------------------
# SINGLE SEQUENCE PREDICTION
# ------------------------------------------
st.header("🔬 Single Peptide Analysis")

seq = st.text_input("Enter peptide sequence")

if st.button("Run Prediction"):
    Xp = extract_features([seq])
    taste = le_taste.inverse_transform(taste_model.predict(Xp))[0]
    sol = le_sol.inverse_transform(sol_model.predict(Xp))[0]
    dock = dock_model.predict(Xp)[0]

    props = biopython_features(seq)

    st.subheader("🧪 Prediction Summary")
    st.write(f"**Taste:** {taste}")
    st.write(f"**Solubility:** {sol}")
    st.write(f"**Docking Score (kcal/mol):** {dock:.3f}")

    st.subheader("📌 Physiochemical & Biochemical Properties")
    for k, v in props.items():
        st.write(f"**{k}:** {round(v, 4)}")

    pdb_text = build_peptide_pdb(seq)
    with open("peptide.pdb", "w") as f:
        f.write(pdb_text)

    st.download_button(
        "Download PDB File",
        pdb_text,
        file_name="peptide.pdb"
    )

    st.subheader("🧬 3D Structure Viewer")
    viewer = show_structure(pdb_text)
    st.components.v1.html(viewer._make_html(), height=500)

# ------------------------------------------
# PDB UPLOAD & VIEW
# ------------------------------------------
st.header("📂 Upload PDB for Visualization")

uploaded_pdb = st.file_uploader("Upload a PDB file", type=["pdb"])
if uploaded_pdb:
    pdb_content = uploaded_pdb.read().decode("utf-8")
    viewer = show_structure(pdb_content)
    st.components.v1.html(viewer._make_html(), height=500)

# ------------------------------------------
# BATCH PREDICTION
# ------------------------------------------
st.header("📊 Batch Prediction")

batch_file = st.file_uploader("Upload CSV/Excel with 'peptide' column")
if batch_file:
    df_batch = pd.read_excel(batch_file) if batch_file.name.endswith("xlsx") else pd.read_csv(batch_file)
    df_batch["peptide"] = df_batch["peptide"].apply(clean_sequence)
    Xb = extract_features(df_batch["peptide"].tolist())

    df_batch["Predicted Taste"] = le_taste.inverse_transform(taste_model.predict(Xb))
    df_batch["Predicted Solubility"] = le_sol.inverse_transform(sol_model.predict(Xb))
    df_batch["Predicted Docking Score"] = dock_model.predict(Xb)

    st.dataframe(df_batch)
    st.download_button(
        "Download Batch Results",
        df_batch.to_csv(index=False),
        "batch_predictions.csv"
    )

# ------------------------------------------
# ANALYTICS & VISUALIZATION
# ------------------------------------------
st.header("📈 Model Analytics")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Taste Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(yt_te, taste_model.predict(X_test)),
                annot=True, fmt="d",
                xticklabels=le_taste.classes_,
                yticklabels=le_taste.classes_, ax=ax)
    st.pyplot(fig)

with col2:
    st.subheader("Solubility Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(ys_te, sol_model.predict(X_test)),
                annot=True, fmt="d",
                xticklabels=le_sol.classes_,
                yticklabels=le_sol.classes_, ax=ax)
    st.pyplot(fig)

st.subheader("PCA Feature Space (Taste)")
coords = PCA(2).fit_transform(X_all)
fig, ax = plt.subplots()
ax.scatter(coords[:, 0], coords[:, 1], c=df["taste"].astype("category").cat.codes, cmap="tab10")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
st.pyplot(fig)

st.subheader("Docking Regression")
fig, ax = plt.subplots()
ax.scatter(yd_te, dock_model.predict(X_test))
ax.plot([yd_te.min(), yd_te.max()], [yd_te.min(), yd_te.max()], "r--")
ax.set_xlabel("Actual")
ax.set_ylabel("Predicted")
st.pyplot(fig)

st.subheader("📌 Overall Metrics")
for k, v in metrics.items():
    st.write(f"**{k}:** {round(v, 4)}")
