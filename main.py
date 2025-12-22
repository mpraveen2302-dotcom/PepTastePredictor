# ==========================================================
# PepTastePredictor – FINAL USER-FRIENDLY PUBLICATION APP
# ==========================================================

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

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="PepTastePredictor",
    layout="wide"
)

# ==========================================================
# CONSTANTS
# ==========================================================

DATASET_PATH = "AIML (4).xlsx"
AA = "ACDEFGHIKLMNPQRSTVWY"

# ==========================================================
# UTILITIES
# ==========================================================

def clean_sequence(seq):
    if not isinstance(seq, str):
        return ""
    seq = seq.upper().replace("\n", "").replace("\t", "").replace(" ", "")
    return "".join([a for a in seq if a in AA])

# ==========================================================
# FEATURE EXTRACTION
# ==========================================================

def aa_composition(seq):
    c = Counter(seq)
    L = len(seq)
    return {f"AA_{a}": c.get(a, 0)/L for a in AA}

def dipeptide_composition(seq):
    dipeptides = ["".join(p) for p in product(AA, repeat=2)]
    counts = Counter(seq[i:i+2] for i in range(len(seq)-1))
    L = max(len(seq)-1, 1)
    return {f"DP_{d}": counts.get(d, 0)/L for d in dipeptides}

def physicochemical_features(seq):
    ana = ProteinAnalysis(seq)
    helix, turn, sheet = ana.secondary_structure_fraction()

    physio = {
        "Length": len(seq),
        "Molecular Weight": round(ana.molecular_weight(), 3),
        "Isoelectric Point (pI)": round(ana.isoelectric_point(), 3),
        "Aromaticity": round(ana.aromaticity(), 3),
        "Instability Index": round(ana.instability_index(), 3),
        "GRAVY": round(ana.gravy(), 3),
        "Net Charge (pH 7)": round(ana.charge_at_pH(7.0), 3),
        "Helix Fraction": round(helix, 3),
        "Turn Fraction": round(turn, 3),
        "Sheet Fraction": round(sheet, 3)
    }

    model_feats = {
        "length": physio["Length"],
        "mw": physio["Molecular Weight"],
        "pI": physio["Isoelectric Point (pI)"],
        "aromaticity": physio["Aromaticity"],
        "instability": physio["Instability Index"],
        "gravy": physio["GRAVY"],
        "charge": physio["Net Charge (pH 7)"],
        "helix": physio["Helix Fraction"],
        "turn": physio["Turn Fraction"],
        "sheet": physio["Sheet Fraction"]
    }

    return physio, model_feats

def extract_features(seqs):
    rows = []
    physio_rows = []

    for s in seqs:
        s = clean_sequence(s)
        if len(s) < 2:
            continue

        physio, model_physio = physicochemical_features(s)
        f = {}
        f.update(model_physio)
        f.update(aa_composition(s))
        f.update(dipeptide_composition(s))

        rows.append(f)
        physio_rows.append(physio)

    return pd.DataFrame(rows).fillna(0), pd.DataFrame(physio_rows)

# ==========================================================
# LOAD & TRAIN MODELS
# ==========================================================

@st.cache_data
def train_models():
    df = pd.read_excel(DATASET_PATH)
    df.columns = df.columns.str.lower().str.strip()

    df["peptide"] = df["peptide"].apply(clean_sequence)
    df = df[df["peptide"].str.len() >= 2]

    X, _ = extract_features(df["peptide"].tolist())

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
        "Docking R2": r2_score(yd_te, dock_model.predict(X_te))
    }

    return taste_model, sol_model, dock_model, le_taste, le_sol, metrics, X, y_taste

taste_model, sol_model, dock_model, le_taste, le_sol, metrics, X_all, y_all = train_models()

# ==========================================================
# PDB & 3D VIEWER
# ==========================================================

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
    view = py3Dmol.view(width=700, height=500)
    view.addModel(pdb_text, "pdb")
    view.setStyle({"cartoon": {"color": "spectrum"}})
    view.zoomTo()
    return view

# ==========================================================
# UI
# ==========================================================

st.title("🧬 PepTastePredictor")
st.markdown("Predict peptide taste, solubility, docking score, and explore structure & analytics.")

# ------------------ SINGLE INPUT ------------------

st.header("🔹 Single Peptide Prediction")

seq = st.text_input("Enter peptide sequence")

if st.button("Predict"):
    seq = clean_sequence(seq)
    Xp, physio = extract_features([seq])

    taste = le_taste.inverse_transform(taste_model.predict(Xp))[0]
    sol = le_sol.inverse_transform(sol_model.predict(Xp))[0]
    dock = dock_model.predict(Xp)[0]

    st.subheader("🧪 Physio-Chemical Properties")
    st.dataframe(physio.T, use_container_width=True)

    st.subheader("🔮 Prediction")
    st.success(f"Taste: {taste}")
    st.info(f"Solubility: {sol}")
    st.warning(f"Docking Score: {dock:.3f} kcal/mol")

    st.subheader("📊 Feature Profile")
    fig, ax = plt.subplots()
    physio.iloc[0].plot(kind="bar", ax=ax)
    st.pyplot(fig)

    pdb = build_pdb(seq)
    st.download_button("⬇ Download PDB", pdb, file_name="peptide.pdb")

    st.subheader("🧬 3D Structure")
    st.components.v1.html(show_structure(pdb)._make_html(), height=500)

# ------------------ PDB UPLOAD ------------------

st.header("🧬 AlphaFold / ColabFold Predicted Structure")

st.markdown(
    "Generate structures using ColabFold and upload the PDB below:\n\n"
    "https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/AlphaFold2_mmseqs2_advanced.ipynb"
)

uploaded_pdb = st.file_uploader("Upload AlphaFold / ColabFold PDB file", type=["pdb"])

if uploaded_pdb:
    pdb_text = uploaded_pdb.read().decode("utf-8")
    st.components.v1.html(show_structure(pdb_text)._make_html(), height=500)
else:
    st.info("No PDB uploaded yet. Use ColabFold to generate one.")

# ------------------ ANALYTICS ------------------

st.header("📈 Model Analytics")

st.write(metrics)

fig, ax = plt.subplots()
coords = PCA(2).fit_transform(X_all)
ax.scatter(coords[:,0], coords[:,1], c=y_all, cmap="tab10")
ax.set_title("PCA – Taste Clustering")
st.pyplot(fig)
