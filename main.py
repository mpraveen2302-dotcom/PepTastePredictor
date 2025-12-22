# ============================================================
# PepTastePredictor – FINAL PUBLICATION-GRADE STREAMLIT APP
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import py3Dmol
import streamlit.components.v1 as components

from collections import Counter
from itertools import product

from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio.PDB import PDBIO
import PeptideBuilder
from PeptideBuilder import Geometry

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score,
    confusion_matrix, mean_squared_error, r2_score
)
from sklearn.decomposition import PCA

# ============================================================
# CONFIG
# ============================================================

st.set_page_config(page_title="PepTastePredictor", layout="wide")
AA = "ACDEFGHIKLMNPQRSTVWY"

# ============================================================
# UTILITIES
# ============================================================

def clean_sequence(seq):
    if not isinstance(seq, str):
        return ""
    seq = seq.upper().replace(" ", "").replace("\n", "").replace("\t", "")
    return "".join([a for a in seq if a in AA])

# ============================================================
# FEATURE EXTRACTION
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

def biopython_features(seq):
    ana = ProteinAnalysis(seq)
    aa_pct = ana.amino_acids_percent
    try:
        instability = ana.instability_index()
    except:
        instability = 0.0

    return {
        "length": len(seq),
        "molecular_weight": ana.molecular_weight(),
        "isoelectric_point": ana.isoelectric_point(),
        "aromaticity": ana.aromaticity(),
        "instability_index": instability,
        "gravy": ana.gravy(),
        "charge_ph7": ana.charge_at_pH(7.0),

        # Secondary structure
        "helix_fraction": ana.secondary_structure_fraction()[0],
        "turn_fraction": ana.secondary_structure_fraction()[1],
        "sheet_fraction": ana.secondary_structure_fraction()[2],

        # Physicochemical composition
        "hydrophobic_fraction": sum(aa_pct[a] for a in "AILMFWV"),
        "polar_fraction": sum(aa_pct[a] for a in "STNQ"),
        "charged_fraction": sum(aa_pct[a] for a in "DEKRH"),
        "aromatic_fraction": sum(aa_pct[a] for a in "FWY"),
    }

def extract_features(seqs):
    rows = []
    for s in seqs:
        f = {}
        f.update(biopython_features(s))
        f.update(aa_composition(s))
        f.update(dipeptide_composition(s))
        rows.append(f)
    return pd.DataFrame(rows).fillna(0)

# ============================================================
# PDB GENERATION (PROPER BACKBONE)
# ============================================================

def build_peptide_pdb(seq):
    structure = PeptideBuilder.initialize_res(seq[0])
    for aa in seq[1:]:
        geom = Geometry.geometry(aa)
        PeptideBuilder.add_residue(structure, geom)

    io = PDBIO()
    io.set_structure(structure)
    pdb_path = "peptide.pdb"
    io.save(pdb_path)

    with open(pdb_path) as f:
        return f.read()

def show_structure(pdb_text):
    view = py3Dmol.view(width=600, height=500)
    view.addModel(pdb_text, "pdb")
    view.setStyle({"cartoon": {"color": "spectrum"}})
    view.zoomTo()
    return view

# ============================================================
# DATASET UPLOAD
# ============================================================

st.sidebar.header("📂 Dataset Upload")
dataset_file = st.sidebar.file_uploader(
    "Upload peptide dataset (Excel)",
    type=["xlsx"]
)

if dataset_file is None:
    st.warning("⬅️ Upload dataset to start analysis")
    st.stop()

# ============================================================
# DATA LOADING & NORMALIZATION
# ============================================================

df = pd.read_excel(dataset_file)
df.columns = (
    df.columns.str.lower()
    .str.strip()
    .str.replace(" ", "_")
    .str.replace("-", "_")
    .str.replace("(", "")
    .str.replace(")", "")
)

df["peptide"] = df["peptide"].apply(clean_sequence)
df = df[df["peptide"].str.len() >= 2].reset_index(drop=True)

# Auto-detect solubility column
for c in ["solubility", "solubilty"]:
    if c in df.columns:
        sol_col = c
        break
else:
    st.error("Solubility column not found")
    st.stop()

# ============================================================
# LABELS
# ============================================================

le_taste = LabelEncoder()
le_sol = LabelEncoder()

y_taste = le_taste.fit_transform(df["taste"].astype(str))
y_sol = le_sol.fit_transform(df[sol_col].astype(str))
y_dock = df["docking_score_kcal/mol"].astype(float)

# ============================================================
# FEATURES
# ============================================================

X = extract_features(df["peptide"].tolist())

# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

strat = y_taste if min(Counter(y_taste).values()) >= 2 else None

X_tr, X_te, yt_tr, yt_te, ys_tr, ys_te, yd_tr, yd_te = train_test_split(
    X, y_taste, y_sol, y_dock,
    test_size=0.2,
    random_state=42,
    stratify=strat
)

# ============================================================
# MODELS
# ============================================================

taste_model = RandomForestClassifier(n_estimators=300, random_state=42)
sol_model = RandomForestClassifier(n_estimators=300, random_state=42)
dock_model = RandomForestRegressor(n_estimators=400, random_state=42)

taste_model.fit(X_tr, yt_tr)
sol_model.fit(X_tr, ys_tr)
dock_model.fit(X_tr, yd_tr)

# ============================================================
# ANALYTICS
# ============================================================

st.title("🧬 PepTastePredictor")

st.header("📊 Model Analytics")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Taste Accuracy", f"{accuracy_score(yt_te, taste_model.predict(X_te)):.3f}")
    st.metric("Taste F1", f"{f1_score(yt_te, taste_model.predict(X_te), average='weighted'):.3f}")

with col2:
    st.metric("Solubility Accuracy", f"{accuracy_score(ys_te, sol_model.predict(X_te)):.3f}")
    st.metric("Solubility F1", f"{f1_score(ys_te, sol_model.predict(X_te), average='weighted'):.3f}")

with col3:
    rmse = np.sqrt(mean_squared_error(yd_te, dock_model.predict(X_te)))
    st.metric("Docking RMSE", f"{rmse:.3f}")
    st.metric("Docking R²", f"{r2_score(yd_te, dock_model.predict(X_te)):.3f}")

# ============================================================
# VISUALIZATIONS (ALL DISTINCT)
# ============================================================

st.header("📈 Visual Analytics")

fig1, ax1 = plt.subplots()
sns.heatmap(confusion_matrix(yt_te, taste_model.predict(X_te)),
            annot=True, fmt="d",
            xticklabels=le_taste.classes_,
            yticklabels=le_taste.classes_, ax=ax1)
ax1.set_title("Taste Confusion Matrix")
st.pyplot(fig1)

fig2, ax2 = plt.subplots()
sns.heatmap(confusion_matrix(ys_te, sol_model.predict(X_te)),
            annot=True, fmt="d",
            xticklabels=le_sol.classes_,
            yticklabels=le_sol.classes_, ax=ax2)
ax2.set_title("Solubility Confusion Matrix")
st.pyplot(fig2)

coords = PCA(2).fit_transform(X)
fig3, ax3 = plt.subplots()
ax3.scatter(coords[:, 0], coords[:, 1], c=y_taste, cmap="tab10")
ax3.set_title("PCA – Taste Clustering")
st.pyplot(fig3)

fig4, ax4 = plt.subplots()
ax4.scatter(yd_te, dock_model.predict(X_te))
ax4.plot([yd_te.min(), yd_te.max()], [yd_te.min(), yd_te.max()], "r--")
ax4.set_title("Docking Regression")
ax4.set_xlabel("Actual")
ax4.set_ylabel("Predicted")
st.pyplot(fig4)

# ============================================================
# PREDICTION & STRUCTURE
# ============================================================

st.header("🔬 Single Peptide Prediction")

seq_input = st.text_input("Enter peptide sequence")

if seq_input:
    seq = clean_sequence(seq_input)
    Xp = extract_features([seq])

    taste = le_taste.inverse_transform(taste_model.predict(Xp))[0]
    sol = le_sol.inverse_transform(sol_model.predict(Xp))[0]
    dock = dock_model.predict(Xp)[0]

    st.json({
        "Sequence": seq,
        "Predicted Taste": taste,
        "Predicted Solubility": sol,
        "Predicted Docking Score": round(float(dock), 3),
        "Total Features Used": X.shape[1]
    })

    pdb_text = build_peptide_pdb(seq)
    viewer = show_structure(pdb_text)
    components.html(viewer._make_html(), height=500)

st.markdown(
    "### AlphaFold / ColabFold\n"
    "https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/AlphaFold2_mmseqs2_advanced.ipynb"
)
