# ==========================================================
# PepTastePredictor — FINAL PUBLICATION VERSION (UPGRADED)
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

st.set_page_config(page_title="PepTastePredictor", layout="wide")

DATASET_PATH = "AIML (4).xlsx"
AA = "ACDEFGHIKLMNPQRSTVWY"

# ==========================================================
# UTILITIES
# ==========================================================

def clean_sequence(seq):
    if not isinstance(seq, str):
        return ""
    seq = seq.upper().replace(" ", "").replace("\n", "").replace("\t", "")
    return "".join([a for a in seq if a in AA])

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

# ==========================================================
# PDB GENERATION + VIEWER
# ==========================================================

def build_peptide_pdb(seq):
    structure = PeptideBuilder.initialize_res(seq[0])
    for aa in seq[1:]:
        geom = Geometry.geometry(aa)
        PeptideBuilder.add_residue(structure, geom)
    io = PDBIO()
    io.set_structure(structure)
    io.save("peptide.pdb")
    return "peptide.pdb"

def show_structure(pdb_text):
    view = py3Dmol.view(width=800, height=500)
    view.addModel(pdb_text, "pdb")
    view.setStyle({"cartoon": {"color": "spectrum"}, "stick": {}})
    view.setBackgroundColor("white")
    view.zoomTo()
    return view

# ==========================================================
# STRUCTURE ANALYSIS (NEW)
# ==========================================================

def ramachandran_from_pdb(pdb_text):
    with open("_tmp.pdb", "w") as f:
        f.write(pdb_text)

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("pep", "_tmp.pdb")
    model = structure[0]

    phi_psi = []
    ppb = PPBuilder()
    for pp in ppb.build_peptides(model):
        for phi, psi in pp.get_phi_psi_list():
            if phi and psi:
                phi_psi.append((np.degrees(phi), np.degrees(psi)))
    return phi_psi

def ca_distance_map(pdb_text):
    with open("_tmp.pdb", "w") as f:
        f.write(pdb_text)

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("pep", "_tmp.pdb")
    cas = [res["CA"].get_vector() for res in structure.get_residues() if "CA" in res]

    n = len(cas)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist[i, j] = (cas[i] - cas[j]).norm()
    return dist

def per_residue_properties(seq):
    hydro = {
        'A':1.8,'C':2.5,'D':-3.5,'E':-3.5,'F':2.8,'G':-0.4,'H':-3.2,
        'I':4.5,'K':-3.9,'L':3.8,'M':1.9,'N':-3.5,'P':-1.6,
        'Q':-3.5,'R':-4.5,'S':-0.8,'T':-0.7,'V':4.2,'W':-0.9,'Y':-1.3
    }
    charge = {'D':-1,'E':-1,'K':1,'R':1,'H':0.1}
    return (
        [hydro.get(a,0) for a in seq],
        [charge.get(a,0) for a in seq]
    )

# ==========================================================
# TRAIN MODELS
# ==========================================================

@st.cache_data
def train_models():
    df = pd.read_excel(DATASET_PATH)
    df.columns = df.columns.str.lower().str.strip()
    df["peptide"] = df["peptide"].apply(clean_sequence)
    df = df[df["peptide"].str.len() >= 2]

    X = build_feature_table(df["peptide"])
    le_taste = LabelEncoder()
    le_sol = LabelEncoder()

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

    metrics = {
        "Taste Accuracy": accuracy_score(yt_te, taste_model.predict(Xte)),
        "Taste F1": f1_score(yt_te, taste_model.predict(Xte), average="weighted"),
        "Solubility Accuracy": accuracy_score(ys_te, sol_model.predict(Xte)),
        "Solubility F1": f1_score(ys_te, sol_model.predict(Xte), average="weighted"),
        "Docking RMSE": np.sqrt(mean_squared_error(yd_te, dock_model.predict(Xte))),
        "Docking R2": r2_score(yd_te, dock_model.predict(Xte))
    }

    return df, X, taste_model, sol_model, dock_model, le_taste, le_sol, metrics

df, X_all, taste_model, sol_model, dock_model, le_taste, le_sol, metrics = train_models()

# ==========================================================
# UI
# ==========================================================

st.title("🧬 PepTastePredictor")

seq = st.text_input("Enter peptide sequence")

if st.button("Predict"):
    seq = clean_sequence(seq)

    feats = physicochemical_features(seq)
    comp = composition_features(seq)
    Xp = build_feature_table([seq])

    taste = le_taste.inverse_transform(taste_model.predict(Xp))[0]
    sol = le_sol.inverse_transform(sol_model.predict(Xp))[0]
    dock = dock_model.predict(Xp)[0]

    st.success(f"**Taste:** {taste}\n\n**Solubility:** {sol}\n\n**Docking score:** {dock:.2f}")

    st.json(feats)
    st.json(comp)

    pdb_file = build_peptide_pdb(seq)
    pdb_text = open(pdb_file).read()

    st.download_button("Download PDB", pdb_text, "peptide.pdb")

    st.components.v1.html(show_structure(pdb_text)._make_html(), height=520)

# ==========================================================
# PDB UPLOAD + ANALYSIS
# ==========================================================

st.markdown("---")
st.markdown("## 🧩 Upload & Analyze PDB")

uploaded = st.file_uploader("Upload PDB file", type=["pdb"])
if uploaded:
    pdb_text = uploaded.read().decode()
    st.components.v1.html(show_structure(pdb_text)._make_html(), height=520)

    st.markdown("### Ramachandran Plot")
    phi_psi = ramachandran_from_pdb(pdb_text)
    if phi_psi:
        phi, psi = zip(*phi_psi)
        fig, ax = plt.subplots()
        ax.scatter(phi, psi, s=20)
        ax.set_xlim(-180,180)
        ax.set_ylim(-180,180)
        st.pyplot(fig)

    st.markdown("### Cα Distance Map")
    dist = ca_distance_map(pdb_text)
    fig, ax = plt.subplots(figsize=(5,5))
    sns.heatmap(dist, cmap="viridis", ax=ax)
    st.pyplot(fig)

# ==========================================================
# ANALYTICS
# ==========================================================

st.markdown("---")
st.markdown("## 📊 Model Analytics")

for k,v in metrics.items():
    st.write(f"- **{k}:** {round(v,3)}")

coords = PCA(2).fit_transform(X_all)
fig, ax = plt.subplots()
ax.scatter(coords[:,0], coords[:,1])
st.pyplot(fig)

st.markdown("### AlphaFold / ColabFold")
st.markdown("https://colab.research.google.com/github/sokrypton/ColabFold")
# ==========================================================
# 📦 BATCH PREDICTION (SAFE ADDITION ONLY)
# ==========================================================

st.markdown("---")
st.markdown("## 📦 Batch Prediction")

batch_file = st.file_uploader(
    "Upload CSV file with a 'peptide' column",
    type=["csv"]
)

if batch_file is not None:
    batch_df = pd.read_csv(batch_file)

    if "peptide" not in batch_df.columns:
        st.error("CSV must contain a column named 'peptide'")
    else:
        # Clean sequences
        batch_df["peptide"] = batch_df["peptide"].apply(clean_sequence)

        # Build features using EXISTING pipeline
        X_batch = build_feature_table(batch_df["peptide"])

        # Predict using EXISTING trained models
        batch_df["Predicted Taste"] = le_taste.inverse_transform(
            taste_model.predict(X_batch)
        )
        batch_df["Predicted Solubility"] = le_sol.inverse_transform(
            sol_model.predict(X_batch)
        )
        batch_df["Predicted Docking Score"] = dock_model.predict(X_batch)

        # Display results
        st.dataframe(batch_df)

        # Download results
        st.download_button(
            label="Download Batch Predictions (CSV)",
            data=batch_df.to_csv(index=False),
            file_name="batch_predictions.csv",
            mime="text/csv"
        )
