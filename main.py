# ==========================================================
# PepTastePredictor – High-Feature Research App (v1.1)
# ==========================================================

import os
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import py3Dmol
from collections import Counter
from itertools import product

from Bio.SeqUtils.ProtParam import ProteinAnalysis

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.decomposition import PCA

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# =========================
# CONFIG
# =========================
DATASET_PATH = "AIML (4).xlsx"
AA = "ACDEFGHIKLMNPQRSTVWY"

st.set_page_config(
    page_title="PepTastePredictor",
    layout="wide",
    page_icon="🧬"
)

# =========================
# SEQUENCE CLEANING
# =========================
def clean_sequence(seq):
    if not isinstance(seq, str):
        return ""
    seq = seq.upper()
    seq = seq.replace("\t", "").replace(" ", "").replace("\n", "")
    return "".join([a for a in seq if a in AA])

# =========================
# FEATURE EXTRACTION
# =========================
def aa_composition(seq):
    c = Counter(seq)
    L = max(len(seq), 1)
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
        "length": len(seq),
        "molecular_weight": ana.molecular_weight(),
        "isoelectric_point": ana.isoelectric_point(),
        "aromaticity": ana.aromaticity(),
        "instability_index": instab,
        "gravy": ana.gravy(),
        "net_charge_pH7": ana.charge_at_pH(7.0),
        "helix_fraction": helix,
        "turn_fraction": turn,
        "sheet_fraction": sheet
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

# =========================
# LOAD & TRAIN MODELS
# =========================
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
        "taste_acc": accuracy_score(yt_te, taste_model.predict(X_te)),
        "taste_f1": f1_score(yt_te, taste_model.predict(X_te), average="weighted"),
        "sol_acc": accuracy_score(ys_te, sol_model.predict(X_te)),
        "sol_f1": f1_score(ys_te, sol_model.predict(X_te), average="weighted"),
        "dock_rmse": np.sqrt(mean_squared_error(yd_te, dock_model.predict(X_te))),
        "dock_r2": r2_score(yd_te, dock_model.predict(X_te))
    }

    return (taste_model, sol_model, dock_model,
            le_taste, le_sol, metrics, X, y_taste)

(taste_model, sol_model, dock_model,
 le_taste, le_sol, metrics, X_all, y_taste_all) = train_models()

# =========================
# VISUALIZATIONS
# =========================
def plot_feature_importance(model, title):
    imp = model.feature_importances_
    idx = np.argsort(imp)[-20:]
    fig, ax = plt.subplots()
    ax.barh(range(len(idx)), imp[idx])
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels(X_all.columns[idx])
    ax.set_title(title)
    return fig

def plot_pca_by_taste():
    coords = PCA(2).fit_transform(X_all)
    fig, ax = plt.subplots()
    sc = ax.scatter(coords[:,0], coords[:,1], c=y_taste_all, cmap="tab10")
    ax.set_title("PCA – Taste Separation")
    plt.colorbar(sc)
    return fig

def plot_sequence_logo(seq):
    freq = Counter(seq)
    fig, ax = plt.subplots()
    ax.bar(freq.keys(), freq.values())
    ax.set_title("Sequence Composition")
    return fig

# =========================
# PDB + 3D VIEWER
# =========================
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

# =========================
# PDF REPORT
# =========================
def generate_pdf(summary):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    text = c.beginText(40, 800)
    for k, v in summary.items():
        text.textLine(f"{k}: {v}")
    c.drawText(text)
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

# =========================
# STREAMLIT UI
# =========================
st.title("🧬 PepTastePredictor")

st.sidebar.header("Input Mode")
mode = st.sidebar.radio("Choose mode", ["Single Peptide", "Batch Prediction"])

if mode == "Single Peptide":
    seq = st.text_input("Enter peptide sequence")

    if st.button("Predict") and seq:
        seq = clean_sequence(seq)
        Xp = extract_features([seq])
        taste = le_taste.inverse_transform(taste_model.predict(Xp))[0]
        sol = le_sol.inverse_transform(sol_model.predict(Xp))[0]
        dock = dock_model.predict(Xp)[0]

        st.subheader("Prediction Summary")
        st.write(f"**Taste:** {taste}")
        st.write(f"**Solubility:** {sol}")
        st.write(f"**Docking Score:** {dock:.3f} kcal/mol")

        st.subheader("Physicochemical Properties")
        props = biopython_features(seq)
        for k, v in props.items():
            st.write(f"- {k.replace('_',' ').title()}: {round(v,3)}")

        st.subheader("Sequence Analytics")
        st.pyplot(plot_sequence_logo(seq))

        pdb_text = build_pdb(seq)
        st.download_button("Download PDB", pdb_text, "peptide.pdb")

        st.subheader("3D Structure")
        st.components.v1.html(show_structure(pdb_text)._make_html(), height=480)

        pdf = generate_pdf({**props, "Taste": taste, "Solubility": sol})
        st.download_button("Download PDF Report", pdf, "analysis_report.pdf")

else:
    file = st.file_uploader("Upload CSV / Excel (column: peptide)")
    if file:
        df = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)
        df["peptide"] = df["peptide"].apply(clean_sequence)
        Xb = extract_features(df["peptide"].tolist())
        df["Taste"] = le_taste.inverse_transform(taste_model.predict(Xb))
        df["Solubility"] = le_sol.inverse_transform(sol_model.predict(Xb))
        df["Docking"] = dock_model.predict(Xb)
        st.dataframe(df)
        st.download_button("Download Results", df.to_csv(index=False), "batch_results.csv")

st.header("📊 Model Analytics")
st.metric("Taste Accuracy", round(metrics["taste_acc"],3))
st.metric("Solubility Accuracy", round(metrics["sol_acc"],3))
st.metric("Docking RMSE", round(metrics["dock_rmse"],3))

st.pyplot(plot_feature_importance(taste_model, "Taste Feature Importance"))
st.pyplot(plot_feature_importance(dock_model, "Docking Feature Importance"))
st.pyplot(plot_pca_by_taste())

st.markdown("### AlphaFold / ColabFold")
st.markdown(
    "Generate high-quality structures here:\n"
    "https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/AlphaFold2_mmseqs2_advanced.ipynb"
)
