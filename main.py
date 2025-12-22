# ==========================================================
# PepTastePredictor
# A COMPLETE END-TO-END PEPTIDE ANALYSIS PLATFORM
# ==========================================================
#
# This application integrates:
#   • Machine Learning (Taste, Solubility, Docking)
#   • Structural Bioinformatics (PDB, RMSD, Ramachandran)
#   • Visualization (3D, PCA, Heatmaps)
#   • Batch Screening
#   • Automated PDF Report Generation
#
# Developed for:
#   • Academic research
#   • iGEM projects
#   • Educational outreach
#   • Officer / reviewer demonstrations
#
# ==========================================================


# ==========================================================
# SECTION 1 — IMPORTS
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
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    r2_score
)
from sklearn.decomposition import PCA

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Image as RLImage,
    Spacer
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4


# ==========================================================
# SECTION 2 — GLOBAL CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="PepTastePredictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

DATASET_PATH = "AIML (4).xlsx"
AA = "ACDEFGHIKLMNPQRSTVWY"


# ==========================================================
# SECTION 3 — FRONTEND STYLING (CSS ONLY)
# ==========================================================

st.markdown("""
<style>

/* Global background */
.stApp {
    background-color: #f4f7fb;
}

/* Titles */
h1, h2, h3 {
    color: #1f3c88;
}

/* Hero banner */
.hero {
    background: linear-gradient(90deg, #1f3c88, #0b7285);
    padding: 30px;
    border-radius: 16px;
    color: white;
    margin-bottom: 30px;
}

/* Card layout */
.card {
    background: white;
    padding: 22px;
    border-radius: 14px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}

/* Metrics highlight */
.metric {
    font-size: 20px;
    font-weight: 600;
    color: #0b7285;
}

/* Footer */
.footer {
    text-align: center;
    color: #6c757d;
    font-size: 13px;
    padding: 30px;
}

</style>
""", unsafe_allow_html=True)


# ==========================================================
# SECTION 4 — SIDEBAR INFORMATION PANEL
# ==========================================================

st.sidebar.image("logo.png", width=120)
st.sidebar.markdown("### PepTastePredictor")
st.sidebar.write("AI-driven peptide analysis platform")
st.sidebar.write("• Taste prediction")
st.sidebar.write("• Solubility prediction")
st.sidebar.write("• Docking estimation")
st.sidebar.write("• Structural bioinformatics")
st.sidebar.write("• Batch screening")
st.sidebar.info("For research & educational use only")


# ==========================================================
# SECTION 5 — SESSION STATE INITIALIZATION (HARD RESET)
# ==========================================================

if "initialized" not in st.session_state:
    st.session_state.initialized = True
    st.session_state.pdb_text = None
    st.session_state.last_prediction = {}
    st.session_state.show_analytics = False



# ==========================================================
# SECTION 6 — UTILITY FUNCTIONS
# ==========================================================

def clean_sequence(seq):
    """
    Cleans peptide sequences by:
    • Uppercasing
    • Removing whitespace
    • Removing invalid amino acids
    """
    if not isinstance(seq, str):
        return ""
    seq = seq.upper().replace(" ", "").replace("\n", "").replace("\t", "")
    return "".join(a for a in seq if a in AA)


# ==========================================================
# SECTION 7 — FEATURE EXTRACTION (SEQUENCE-BASED)
# ==========================================================

def physicochemical_features(seq):
    """
    Computes physicochemical properties using Biopython:
    • Molecular weight
    • pI
    • GRAVY
    • Instability
    • Secondary structure fractions
    """
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
    """
    Calculates amino acid composition groups:
    • Hydrophobic
    • Polar
    • Charged
    • Aromatic
    """
    c = Counter(seq)
    L = len(seq)

    return {
        "Hydrophobic (%)": round(100 * sum(c[a] for a in "AILMFWV") / L, 1),
        "Polar (%)": round(100 * sum(c[a] for a in "STNQ") / L, 1),
        "Charged (%)": round(100 * sum(c[a] for a in "DEKRH") / L, 1),
        "Aromatic (%)": round(100 * sum(c[a] for a in "FWY") / L, 1),
    }


def model_features(seq):
    """
    Converts peptide sequence into ML-compatible feature vector.
    Includes:
    • Global physicochemical descriptors
    • Normalized amino acid frequencies
    """
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
    """
    Builds a pandas DataFrame of ML features for a list of sequences.
    """
    return pd.DataFrame([model_features(s) for s in seqs]).fillna(0)


# ==========================================================
# SECTION 8 — STRUCTURE GENERATION & ANALYSIS
# ==========================================================

def build_peptide_pdb(seq):
    """
    Generates a peptide PDB structure using PeptideBuilder.
    """
    structure = PeptideBuilder.initialize_res(seq[0])

    for aa in seq[1:]:
        PeptideBuilder.add_residue(structure, Geometry.geometry(aa))

    io = PDBIO()
    io.set_structure(structure)
    io.save("predicted_peptide.pdb")

    return open("predicted_peptide.pdb").read()


def show_structure(pdb_text):
    """
    Displays a 3D structure using py3Dmol.
    """
    view = py3Dmol.view(width=800, height=450)
    view.addModel(pdb_text, "pdb")
    view.setStyle({"cartoon": {"color": "spectrum"}})
    view.zoomTo()
    return view


def ramachandran(pdb_text):
    """
    Computes phi-psi angles for Ramachandran plot.
    """
    open("_tmp.pdb", "w").write(pdb_text)
    structure = PDBParser(QUIET=True).get_structure("x", "_tmp.pdb")[0]

    points = []
    for pp in PPBuilder().build_peptides(structure):
        for phi, psi in pp.get_phi_psi_list():
            if phi and psi:
                points.append((np.degrees(phi), np.degrees(psi)))

    return points


def ca_distance_map(pdb_text):
    """
    Computes Cα distance matrix.
    """
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
    """
    Computes Cα RMSD relative to first residue.
    """
    open("_rmsd.pdb", "w").write(pdb_text)
    structure = PDBParser(QUIET=True).get_structure("x", "_rmsd.pdb")

    cas = [r["CA"].get_vector() for r in structure.get_residues() if "CA" in r]
    if len(cas) < 2:
        return None

    ref = cas[0]
    return np.sqrt(np.mean([(v - ref).norm()**2 for v in cas]))


# ==========================================================
# SECTION 9 — MODEL TRAINING
# ==========================================================

@st.cache_data
def train_models():
    """
    Trains Random Forest models for:
    • Taste classification
    • Solubility classification
    • Docking score regression
    """

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
        X, y_taste, y_sol, y_dock,
        test_size=0.2,
        random_state=42
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


# ==========================================================
# SECTION 10 — LOAD MODELS
# ==========================================================

X_all, taste_model, sol_model, dock_model, le_taste, le_sol, metrics = train_models()


# ==========================================================
# SECTION 11 — HERO HEADER
# ==========================================================
# ==========================================================
# SECTION X — PDF REPORT GENERATION
# ==========================================================

def generate_pdf(metrics, prediction, image_paths):
    """
    Generates a full PDF report containing:
    - Model performance metrics
    - Prediction results
    - Embedded structural plots
    """

    file_name = "PepTastePredictor_Full_Report.pdf"
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(file_name, pagesize=A4)

    story = []

    # Title
    story.append(Paragraph("<b>PepTastePredictor</b>", styles["Title"]))
    story.append(Spacer(1, 12))

    # Overview
    story.append(Paragraph(
        "An AI-driven platform for peptide taste, solubility, docking, "
        "and structural bioinformatics analysis.",
        styles["Normal"]
    ))
    story.append(Spacer(1, 12))

    # Model metrics
    story.append(Paragraph("<b>Model Performance Metrics</b>", styles["Heading2"]))
    for k, v in metrics.items():
        story.append(Paragraph(f"{k}: {round(v,4)}", styles["Normal"]))
    story.append(Spacer(1, 12))

    # Prediction results
    story.append(Paragraph("<b>Prediction Results</b>", styles["Heading2"]))
    for k, v in prediction.items():
        story.append(Paragraph(f"{k}: {v}", styles["Normal"]))
    story.append(Spacer(1, 12))

    # Structural analysis
    story.append(Paragraph("<b>Structural Analysis</b>", styles["Heading2"]))
    story.append(Paragraph(
        "Structural analysis includes 3D visualization, Ramachandran plot, "
        "Cα distance mapping, and RMSD-based deviation assessment.",
        styles["Normal"]
    ))
    story.append(Spacer(1, 12))

    # Add figures
    for img in image_paths:
        if os.path.exists(img):
            story.append(RLImage(img, width=400, height=300))
            story.append(Spacer(1, 12))

    # Build PDF
    doc.build(story)

    return file_name

st.markdown("""
<div class="hero">
<h1>🧬 PepTastePredictor</h1>
<p>
An integrated machine learning and structural bioinformatics platform
for peptide taste, solubility, docking, and structural analysis.
</p>
</div>
""", unsafe_allow_html=True)
# ==========================================================
# SECTION 12 — MODE SELECTION
# ==========================================================

st.markdown("## 🔧 Prediction & Analysis Mode Selection")

mode = st.radio(
    "Choose the analysis mode",
    [
        "Single Peptide Prediction",
        "Batch Peptide Prediction",
        "PDB Upload & Structural Analysis"
    ],
    horizontal=True
)


# ==========================================================
# SECTION 13 — SINGLE PEPTIDE PREDICTION MODE
# ==========================================================

if mode == "Single Peptide Prediction":

    st.markdown("## 🔬 Single Peptide Prediction")

    seq = st.text_input(
        "Enter peptide sequence (FASTA single-letter code)",
        help="Example: AGLWFK"
    )

    if st.button("Run Prediction"):

        # --------------------------------------------------
        # Sequence cleaning & feature generation
        # --------------------------------------------------
        seq = clean_sequence(seq)
        Xp = pd.DataFrame([model_features(seq)])

        # --------------------------------------------------
        # ML Predictions
        # --------------------------------------------------
        taste = le_taste.inverse_transform(taste_model.predict(Xp))[0]
        sol = le_sol.inverse_transform(sol_model.predict(Xp))[0]
        dock = dock_model.predict(Xp)[0]

        st.session_state.last_prediction = {
            "Sequence": seq,
            "Predicted taste": taste,
            "Predicted solubility": sol,
            "Docking score (kcal/mol)": round(dock, 3)
        }

        # 🔑 Enable analytics AFTER prediction
        st.session_state.show_analytics = True

        # --------------------------------------------------
        # Display prediction summary
        # --------------------------------------------------
        st.markdown(f"""
        <div class="card">
            <div class="metric">Taste Prediction</div>
            <p>{taste}</p>
            <div class="metric">Solubility Prediction</div>
            <p>{sol}</p>
            <div class="metric">Docking Score</div>
            <p>{dock:.3f} kcal/mol</p>
        </div>
        """, unsafe_allow_html=True)

        # --------------------------------------------------
        # Physicochemical properties
        # --------------------------------------------------
        st.markdown("### 📌 Physicochemical Properties")
        for k, v in physicochemical_features(seq).items():
            st.write(f"{k}: {v}")

        # --------------------------------------------------
        # Composition analysis
        # --------------------------------------------------
        st.markdown("### 🧪 Amino Acid Composition Summary")
        for k, v in composition_features(seq).items():
            st.write(f"{k}: {v}")

        # --------------------------------------------------
        # Structure generation
        # --------------------------------------------------
        st.markdown("## 🧬 Predicted 3D Peptide Structure")

        pdb_text = build_peptide_pdb(seq)
        st.session_state.pdb_text = pdb_text

        st.download_button(
            "⬇️ Download Predicted PDB",
            pdb_text,
            file_name="predicted_peptide.pdb"
        )

        st.components.v1.html(
            show_structure(pdb_text)._make_html(),
            height=520
        )

        # --------------------------------------------------
        # RMSD analysis
        # --------------------------------------------------
        rmsd_val = ca_rmsd(pdb_text)
        if rmsd_val:
            st.success(f"Cα RMSD (structural deviation): {rmsd_val:.3f} Å")

        # --------------------------------------------------
        # Ramachandran plot
        # --------------------------------------------------
        st.markdown("### 📐 Ramachandran Plot")

        phi_psi = ramachandran(pdb_text)
        if phi_psi:
            phi, psi = zip(*phi_psi)
            fig_rama, ax_rama = plt.subplots()
            ax_rama.scatter(phi, psi, s=20)
            ax_rama.set_xlabel("Phi (°)")
            ax_rama.set_ylabel("Psi (°)")
            ax_rama.set_title("Ramachandran Plot")
            fig_rama.savefig("rama.png")
            st.pyplot(fig_rama)

        # --------------------------------------------------
        # Cα distance map
        # --------------------------------------------------
        st.markdown("### 🗺️ Cα Distance Map")

        dist_map = ca_distance_map(pdb_text)
        fig_dist, ax_dist = plt.subplots(figsize=(5, 5))
        sns.heatmap(dist_map, cmap="viridis", ax=ax_dist)
        ax_dist.set_title("Cα Distance Heatmap")
        fig_dist.savefig("dist.png")
        st.pyplot(fig_dist)

        # --------------------------------------------------
        # PDF report generation
        # --------------------------------------------------
        st.markdown("## 📄 Full PDF Report")

        pdf_path = generate_pdf(
            metrics,
            st.session_state.last_prediction,
            ["rama.png", "dist.png"]
        )

        with open(pdf_path, "rb") as f:
            st.download_button(
                "📥 Download Complete PDF Report",
                f,
                file_name="PepTastePredictor_Full_Report.pdf",
                mime="application/pdf"
            )

# ==========================================================
# SECTION 14 — BATCH PEPTIDE PREDICTION MODE
# ==========================================================

if mode == "Batch Peptide Prediction":

    st.markdown("## 📦 Batch Peptide Prediction")

    batch_file = st.file_uploader(
        "Upload CSV file containing a 'peptide' column",
        type=["csv"]
    )

    if batch_file is not None:

        batch_df = pd.read_csv(batch_file)

        if "peptide" not in batch_df.columns:
            st.error("CSV file must contain a column named 'peptide'")
        else:
            batch_df["peptide"] = batch_df["peptide"].apply(clean_sequence)

            X_batch = build_feature_table(batch_df["peptide"])

            batch_df["Predicted Taste"] = le_taste.inverse_transform(
                taste_model.predict(X_batch)
            )
            batch_df["Predicted Solubility"] = le_sol.inverse_transform(
                sol_model.predict(X_batch)
            )
            batch_df["Predicted Docking Score"] = dock_model.predict(X_batch)

            st.markdown("### ✅ Batch Prediction Results")
            st.dataframe(batch_df)

            st.download_button(
                "⬇️ Download Batch Results",
                batch_df.to_csv(index=False),
                file_name="batch_predictions.csv"
            )

            # 🔑 Enable analytics after batch
            st.session_state.show_analytics = True

# ==========================================================
# SECTION 15 — PDB UPLOAD & STRUCTURAL ANALYSIS MODE
# ==========================================================

if mode == "PDB Upload & Structural Analysis":

    st.markdown("## 🧩 Upload & Analyze PDB Structure")

    uploaded_pdb = st.file_uploader(
        "Upload a PDB file for structural analysis",
        type=["pdb"]
    )

    if uploaded_pdb is not None:

        pdb_text = uploaded_pdb.read().decode()
        st.session_state.pdb_text = pdb_text

        # 🔑 Enable analytics after upload
        st.session_state.show_analytics = True

        # --------------------------------------------------
        # 3D structure visualization
        # --------------------------------------------------
        st.markdown("### 🧬 3D Structure Viewer")

        st.components.v1.html(
            show_structure(st.session_state.pdb_text)._make_html(),
            height=520
        )

        # --------------------------------------------------
        # RMSD
        # --------------------------------------------------
        rmsd_val = ca_rmsd(st.session_state.pdb_text)
        if rmsd_val:
            st.success(f"Cα RMSD (structural deviation): {rmsd_val:.3f} Å")

        # --------------------------------------------------
        # Ramachandran plot
        # --------------------------------------------------
        st.markdown("### 📐 Ramachandran Plot")

        phi_psi = ramachandran(st.session_state.pdb_text)
        if phi_psi:
            phi, psi = zip(*phi_psi)
            fig, ax = plt.subplots()
            ax.scatter(phi, psi)
            ax.set_xlabel("Phi (°)")
            ax.set_ylabel("Psi (°)")
            ax.set_title("Ramachandran Plot")
            st.pyplot(fig)

        # --------------------------------------------------
        # Cα distance map
        # --------------------------------------------------
        st.markdown("### 🗺️ Cα Distance Map")

        fig, ax = plt.subplots(figsize=(5, 5))
        sns.heatmap(
            ca_distance_map(st.session_state.pdb_text),
            cmap="viridis",
            ax=ax
        )
        ax.set_title("Cα Distance Heatmap")
        st.pyplot(fig)

with st.expander("📊 Model Performance & Dataset Analytics"):

# ==========================================================
# SECTION 16 — MODEL & DATASET ANALYTICS (POST-ACTION ONLY)
# ==========================================================

if st.session_state.show_analytics is True:

    with st.expander("📊 Model Performance & Dataset Analytics"):

        st.markdown("### Model Performance Metrics")

        for k, v in metrics.items():
            st.write(f"{k}: {round(v, 4)}")

        st.markdown("### Dataset Feature Space (PCA Projection)")

        coords = PCA(2).fit_transform(X_all)

        fig_pca, ax_pca = plt.subplots()
        ax_pca.scatter(coords[:, 0], coords[:, 1], alpha=0.6)
        ax_pca.set_xlabel("PC1")
        ax_pca.set_ylabel("PC2")
        ax_pca.set_title("PCA of Peptide Feature Space")

        st.pyplot(fig_pca)



    # ==========================================================
    # SECTION 17 — DATASET FEATURE SPACE (PCA)
    # ==========================================================

    st.markdown("## 🧠 Dataset Feature Space (PCA Projection)")

    coords = PCA(2).fit_transform(X_all)

    fig_pca, ax_pca = plt.subplots()
    ax_pca.scatter(coords[:, 0], coords[:, 1], alpha=0.6)
    ax_pca.set_xlabel("PC1")
    ax_pca.set_ylabel("PC2")
    ax_pca.set_title("PCA of Peptide Feature Space")

    st.pyplot(fig_pca)



# ==========================================================
# SECTION 18 — FOOTER
# ==========================================================

st.markdown("""
<div class="footer">
© 2025 <b>PepTastePredictor</b><br>
An AI + Structural Bioinformatics platform for peptide analysis<br>
For academic, educational, and research use
</div>
""", unsafe_allow_html=True)
