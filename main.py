# ============================
# PepTastePredictor – FINAL
# ============================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import py3Dmol
import streamlit.components.v1 as components

from collections import Counter
from itertools import product

from Bio.SeqUtils.ProtParam import ProteinAnalysis

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, mean_squared_error, r2_score
from sklearn.decomposition import PCA

# ============================
# CONFIG
# ============================

DATASET_PATH = "AIML (4).xlsx"
AA = "ACDEFGHIKLMNPQRSTVWY"

st.set_page_config(
    page_title="PepTastePredictor",
    layout="wide"
)

# ============================
# SEQUENCE CLEANING
# ============================

def clean_sequence(seq):
    if not isinstance(seq, str):
        return ""
    seq = seq.upper()
    seq = seq.replace("\t", "").replace(" ", "").replace("\n", "")
    seq = "".join([a for a in seq if a in AA])
    return seq

# ============================
# FEATURE EXTRACTION
# ============================

def aa_composition(seq):
    c = Counter(seq)
    L = len(seq)
    return {f"AA_{a}": c.get(a, 0) / L for a in AA}

def dipeptide_composition(seq):
    dipeptides = ["".join(p) for p in product(AA, repeat=2)]
    counts = Counter(seq[i:i+2] for i in range(len(seq)-1))
    L = max(len(seq)-1, 1)
    return {f"DP_{d}": counts.get(d, 0) / L for d in dipeptides}

def biopython_features(seq):
    ana = ProteinAnalysis(seq)
    helix, turn, sheet = ana.secondary_structure_fraction()
    try:
        instab = ana.instability_index()
    except:
        instab = 0.0

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
    feats = []
    for s in seqs:
        f = {}
        f.update(biopython_features(s))
        f.update(aa_composition(s))
        f.update(dipeptide_composition(s))
        feats.append(f)
    return pd.DataFrame(feats).fillna(0)

# ============================
# LOAD & TRAIN MODELS
# ============================

@st.cache_resource
def train_models():
    df = pd.read_excel(DATASET_PATH)
    df.columns = df.columns.str.lower().str.strip()
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
        X, y_taste, y_sol, y_dock, test_size=0.2, random_state=42, stratify=strat
    )

    taste_model = RandomForestClassifier(n_estimators=300, random_state=42)
    sol_model = RandomForestClassifier(n_estimators=300, random_state=42)
    dock_model = RandomForestRegressor(n_estimators=400, random_state=42)

    taste_model.fit(X_tr, yt_tr)
    sol_model.fit(X_tr, ys_tr)
    dock_model.fit(X_tr, yd_tr)

    metrics = {
        "taste_acc": accuracy_score(yt_te, taste_model.predict(X_te)),
        "taste_f1": f1_score(yt_te, taste_model.predict(X_te), average="weighted"),
        "sol_acc": accuracy_score(ys_te, sol_model.predict(X_te)),
        "sol_f1": f1_score(ys_te, sol_model.predict(X_te), average="weighted"),
        "dock_rmse": np.sqrt(mean_squared_error(yd_te, dock_model.predict(X_te))),
        "dock_r2": r2_score(yd_te, dock_model.predict(X_te))
    }

    return (
        X, X_te, yt_te, ys_te, yd_te,
        taste_model, sol_model, dock_model,
        le_taste, le_sol, metrics
    )

(
    X_all, X_test, y_t_test, y_s_test, y_d_test,
    taste_model, sol_model, dock_model,
    le_taste, le_sol, metrics
) = train_models()

# ============================
# PDB BUILD (CA TRACE)
# ============================

def build_ca_pdb(seq):
    lines = []
    x = 0.0
    for i, aa in enumerate(seq, start=1):
        lines.append(
            f"ATOM  {i:5d}  CA  {aa:>3s} A{i:4d}    {x:8.3f}{0.000:8.3f}{0.000:8.3f}  1.00  0.00           C"
        )
        x += 3.8
    lines.append("TER")
    lines.append("END")
    return "\n".join(lines)

# ============================
# 3D VIEWER (FIXED)
# ============================

def show_structure(pdb_text):
    view = py3Dmol.view(width=800, height=500)
    view.addModel(pdb_text, "pdb")
    view.setStyle({"cartoon": {"color": "spectrum"}})
    view.setBackgroundColor("white")
    view.zoomTo()
    return view

# ============================
# UI
# ============================

st.title("🧬 PepTastePredictor")
st.markdown("Predict peptide **taste**, **solubility**, **docking score**, explore **features**, **analytics**, and **3D structure**.")

# ---------- INPUT ----------
seq = st.text_input("Enter Peptide Sequence")

uploaded_pdb = st.file_uploader("Upload AlphaFold / ColabFold PDB (optional)", type=["pdb"])

# ---------- PREDICTION ----------
if st.button("Predict") and seq:
    seq = clean_sequence(seq)
    Xp = extract_features([seq])

    taste = le_taste.inverse_transform(taste_model.predict(Xp))[0]
    sol = le_sol.inverse_transform(sol_model.predict(Xp))[0]
    dock = dock_model.predict(Xp)[0]

    bio = biopython_features(seq)

    st.subheader("🔬 Prediction Summary")
    st.write(f"**Taste:** {taste}")
    st.write(f"**Solubility:** {sol}")
    st.write(f"**Docking Score:** {dock:.3f} kcal/mol")

    st.subheader("🧪 Physico-Chemical Properties")
    for k, v in bio.items():
        st.write(f"**{k.replace('_',' ').title()}**: {round(v,4)}")

    pdb_text = build_ca_pdb(seq)
    st.download_button("⬇ Download PDB", pdb_text, "peptide.pdb")

    st.subheader("🧬 3D Structure Viewer")
    viewer = show_structure(pdb_text)
    components.html(viewer._make_html(), height=500)

# ---------- UPLOADED PDB VIEW ----------
if uploaded_pdb is not None:
    pdb_content = uploaded_pdb.read().decode("utf-8")
    st.subheader("🧬 Uploaded Structure Viewer")
    viewer = show_structure(pdb_content)
    components.html(viewer._make_html(), height=500)

# ============================
# ANALYTICS
# ============================

st.header("📊 Model Analytics")

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_t_test, taste_model.predict(X_test)),
                annot=True, fmt="d", ax=ax)
    ax.set_title("Taste Confusion Matrix")
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_s_test, sol_model.predict(X_test)),
                annot=True, fmt="d", ax=ax)
    ax.set_title("Solubility Confusion Matrix")
    st.pyplot(fig)

fig, ax = plt.subplots()
coords = PCA(2).fit_transform(X_all)
ax.scatter(coords[:,0], coords[:,1], alpha=0.6)
ax.set_title("PCA – Feature Space")
st.pyplot(fig)

st.subheader("📈 Performance Metrics")
st.write(metrics)

st.markdown(
    "### 🔗 AlphaFold / ColabFold\n"
    "https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/AlphaFold2_mmseqs2_advanced.ipynb"
)
