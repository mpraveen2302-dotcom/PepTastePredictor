# ==========================================================
# PepTastePredictor — FINAL COMPLETE VERSION (PDF + ANALYSIS)
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import py3Dmol
import os

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

from reportlab.platypus import SimpleDocTemplate, Paragraph, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

# ==========================================================
# CONFIG
# ==========================================================

st.set_page_config(page_title="PepTastePredictor", layout="wide")

DATASET_PATH = "AIML (4).xlsx"
AA = "ACDEFGHIKLMNPQRSTVWY"

# ==========================================================
# SESSION STATE
# ==========================================================

if "pdb_text" not in st.session_state:
    st.session_state.pdb_text = None
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = {}

# ==========================================================
# UTILITIES
# ==========================================================

def clean_sequence(seq):
    if not isinstance(seq, str):
        return ""
    seq = seq.upper().replace(" ", "").replace("\n", "")
    return "".join(a for a in seq if a in AA)

# ==========================================================
# FEATURES
# ==========================================================

def physicochemical_features(seq):
    ana = ProteinAnalysis(seq)
    h, t, s = ana.secondary_structure_fraction()
    return {
        "Length": len(seq),
        "Molecular weight (Da)": round(ana.molecular_weight(), 2),
        "Isoelectric point": round(ana.isoelectric_point(), 2),
        "Net charge (pH 7)": round(ana.charge_at_pH(7.0), 2),
        "Aromaticity": round(ana.aromaticity(), 3),
        "GRAVY": round(ana.gravy(), 3),
        "Instability index": round(ana.instability_index(), 2),
        "Helix fraction": round(h, 3),
        "Turn fraction": round(t, 3),
        "Sheet fraction": round(s, 3),
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
# STRUCTURE + ANALYSIS
# ==========================================================

def build_peptide_pdb(seq):
    structure = PeptideBuilder.initialize_res(seq[0])
    for aa in seq[1:]:
        PeptideBuilder.add_residue(structure, Geometry.geometry(aa))
    io = PDBIO()
    io.set_structure(structure)
    io.save("peptide.pdb")
    return open("peptide.pdb").read()

def show_structure(pdb_text):
    view = py3Dmol.view(width=800, height=450)
    view.addModel(pdb_text, "pdb")
    view.setStyle({"cartoon": {"color": "spectrum"}})
    view.zoomTo()
    return view

def ramachandran(pdb_text):
    open("_tmp.pdb", "w").write(pdb_text)
    structure = PDBParser(QUIET=True).get_structure("x", "_tmp.pdb")[0]
    pts = []
    for pp in PPBuilder().build_peptides(structure):
        for phi, psi in pp.get_phi_psi_list():
            if phi and psi:
                pts.append((np.degrees(phi), np.degrees(psi)))
    return pts

def ca_distance_map(pdb_text):
    open("_tmp.pdb", "w").write(pdb_text)
    structure = PDBParser(QUIET=True).get_structure("x", "_tmp.pdb")
    cas = [r["CA"].get_vector() for r in structure.get_residues() if "CA" in r]
    n = len(cas)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            mat[i, j] = (cas[i] - cas[j]).norm()
    return mat

def ca_rmsd(pdb_text):
    open("_rmsd.pdb", "w").write(pdb_text)
    structure = PDBParser(QUIET=True).get_structure("x", "_rmsd.pdb")
    cas = [r["CA"].get_vector() for r in structure.get_residues() if "CA" in r]
    if len(cas) < 2:
        return None
    ref = cas[0]
    return np.sqrt(np.mean([(v - ref).norm()**2 for v in cas]))

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

    metrics = {
        "Taste accuracy": accuracy_score(yt_te, taste_model.predict(Xte)),
        "Taste F1": f1_score(yt_te, taste_model.predict(Xte), average="weighted"),
        "Solubility accuracy": accuracy_score(ys_te, sol_model.predict(Xte)),
        "Solubility F1": f1_score(ys_te, sol_model.predict(Xte), average="weighted"),
        "Docking RMSE": np.sqrt(mean_squared_error(yd_te, dock_model.predict(Xte))),
        "Docking R²": r2_score(yd_te, dock_model.predict(Xte)),
    }

    return X, taste_model, sol_model, dock_model, le_taste, le_sol, metrics

X_all, taste_model, sol_model, dock_model, le_taste, le_sol, metrics = train_models()

# ==========================================================
# PDF REPORT
# ==========================================================

def generate_pdf(metrics, prediction, fig_paths):
    file = "PepTastePredictor_Full_Report.pdf"
    doc = SimpleDocTemplate(file, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("<b>PepTastePredictor</b>", styles["Title"]))
    story.append(Paragraph("End-to-end peptide analysis report", styles["Normal"]))

    story.append(Paragraph("<b>Model Performance</b>", styles["Heading2"]))
    for k, v in metrics.items():
        story.append(Paragraph(f"{k}: {round(v,4)}", styles["Normal"]))

    story.append(Paragraph("<b>Prediction Results</b>", styles["Heading2"]))
    for k, v in prediction.items():
        story.append(Paragraph(f"{k}: {v}", styles["Normal"]))

    story.append(Paragraph("<b>Structural Analysis</b>", styles["Heading2"]))
    for p in fig_paths:
        story.append(RLImage(p, width=400, height=300))

    doc.build(story)
    return file

# ==========================================================
# UI
# ==========================================================

st.title("🧬 PepTastePredictor")

mode = st.radio("Mode", ["Single peptide", "Batch peptides", "PDB analysis"], horizontal=True)

# ---------------- SINGLE ----------------

if mode == "Single peptide":

    seq = st.text_input("Enter peptide sequence")

    if st.button("Predict"):
        seq = clean_sequence(seq)
        Xp = pd.DataFrame([model_features(seq)])

        taste = le_taste.inverse_transform(taste_model.predict(Xp))[0]
        sol = le_sol.inverse_transform(sol_model.predict(Xp))[0]
        dock = dock_model.predict(Xp)[0]

        st.session_state.last_prediction = {
            "Sequence": seq,
            "Taste": taste,
            "Solubility": sol,
            "Docking score": round(dock, 2)
        }

        st.success(st.session_state.last_prediction)

        pdb = build_peptide_pdb(seq)
        st.session_state.pdb_text = pdb

        st.download_button("Download PDB", pdb, "predicted_peptide.pdb")
        st.components.v1.html(show_structure(pdb)._make_html(), height=500)

        rmsd = ca_rmsd(pdb)
        st.write(f"Cα RMSD: {round(rmsd,3)} Å")

        # Plots
        phi, psi = zip(*ramachandran(pdb))
        fig1, ax1 = plt.subplots()
        ax1.scatter(phi, psi)
        fig1.savefig("rama.png")
        st.pyplot(fig1)

        fig2, ax2 = plt.subplots(figsize=(5,5))
        sns.heatmap(ca_distance_map(pdb), ax=ax2)
        fig2.savefig("dist.png")
        st.pyplot(fig2)

        # PDF
        pdf = generate_pdf(metrics, st.session_state.last_prediction, ["rama.png", "dist.png"])
        st.download_button("Download Full PDF Report", open(pdf, "rb"), pdf)

# ---------------- BATCH ----------------

if mode == "Batch peptides":
    batch = st.file_uploader("Upload CSV", type=["csv"])
    if batch:
        df = pd.read_csv(batch)
        df["peptide"] = df["peptide"].apply(clean_sequence)
        Xb = build_feature_table(df["peptide"])

        df["Taste"] = le_taste.inverse_transform(taste_model.predict(Xb))
        df["Solubility"] = le_sol.inverse_transform(sol_model.predict(Xb))
        df["Docking"] = dock_model.predict(Xb)

        st.dataframe(df)
        st.download_button("Download results", df.to_csv(index=False), "batch_results.csv")

# ---------------- PDB ANALYSIS ----------------

if mode == "PDB analysis":
    up = st.file_uploader("Upload PDB", type=["pdb"])
    if up:
        pdb = up.read().decode()
        st.components.v1.html(show_structure(pdb)._make_html(), height=500)

        rmsd = ca_rmsd(pdb)
        st.write(f"Cα RMSD: {round(rmsd,3)} Å")

        phi, psi = zip(*ramachandran(pdb))
        fig, ax = plt.subplots()
        ax.scatter(phi, psi)
        st.pyplot(fig)

        fig, ax = plt.subplots(figsize=(5,5))
        sns.heatmap(ca_distance_map(pdb), ax=ax)
        st.pyplot(fig)

# ---------------- METRICS ----------------

st.markdown("## 📊 Model Metrics")
for k, v in metrics.items():
    st.write(f"{k}: {round(v,4)}")
