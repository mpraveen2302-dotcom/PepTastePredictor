# ==========================================================
# PepTastePredictor — FINAL PUBLICATION VERSION (STABLE)
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import py3Dmol

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

# ==========================================================
# CONFIG
# ==========================================================

st.set_page_config(page_title="PepTastePredictor", layout="wide")

DATASET_PATH = "AIML (4).xlsx"
AA = "ACDEFGHIKLMNPQRSTVWY"

# ==========================================================
# UI STYLING (SAFE)
# ==========================================================

st.markdown("""
<style>
.stApp { background-color: #f6f8fc; }
h1, h2, h3 { color: #1f3c88; }

.card {
    background: white;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    margin-bottom: 16px;
}

.metric { font-size: 18px; font-weight: 600; color: #0b7285; }

.footer {
    text-align: center;
    color: #6c757d;
    font-size: 13px;
    padding: 20px;
}
</style>
""", unsafe_allow_html=True)

# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title("🧬 PepTastePredictor")
st.sidebar.write("AI-powered peptide taste & solubility prediction")
st.sidebar.write("• Single prediction")
st.sidebar.write("• Batch prediction")
st.sidebar.write("• 3D structure analysis")
st.sidebar.write("• Academic & research use")

# ==========================================================
# SESSION STATE
# ==========================================================

if "pdb_text" not in st.session_state:
    st.session_state.pdb_text = None

# ==========================================================
# UTILITIES
# ==========================================================

def clean_sequence(seq):
    if not isinstance(seq, str):
        return ""
    seq = seq.upper().replace(" ", "").replace("\n", "").replace("\t", "")
    return "".join(a for a in seq if a in AA)

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
# PDB FUNCTIONS
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
    view.setBackgroundColor("white")
    view.zoomTo()
    return view

def ramachandran_from_pdb(pdb_text):
    open("_tmp.pdb", "w").write(pdb_text)
    structure = PDBParser(QUIET=True).get_structure("x", "_tmp.pdb")[0]
    out = []
    for pp in PPBuilder().build_peptides(structure):
        for phi, psi in pp.get_phi_psi_list():
            if phi and psi:
                out.append((np.degrees(phi), np.degrees(psi)))
    return out

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

# ==========================================================
# TRAIN MODELS (UNCHANGED)
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
        "Taste F1 score": f1_score(yt_te, taste_model.predict(Xte), average="weighted"),
        "Solubility accuracy": accuracy_score(ys_te, sol_model.predict(Xte)),
        "Solubility F1 score": f1_score(ys_te, sol_model.predict(Xte), average="weighted"),
        "Docking RMSE": np.sqrt(mean_squared_error(yd_te, dock_model.predict(Xte))),
        "Docking R²": r2_score(yd_te, dock_model.predict(Xte))
    }

    return df, X, taste_model, sol_model, dock_model, le_taste, le_sol, metrics

df, X_all, taste_model, sol_model, dock_model, le_taste, le_sol, metrics = train_models()

# ==========================================================
# HEADER
# ==========================================================

st.markdown("# 🧬 PepTastePredictor")
st.write("AI-powered peptide taste, solubility & docking prediction with structural insights")

# ==========================================================
# SINGLE PREDICTION
# ==========================================================

st.markdown("## 🔬 Single Peptide Prediction")

seq = st.text_input("Enter peptide sequence", key="single_seq")

if st.button("Predict", key="single_predict"):
    seq = clean_sequence(seq)
    Xp = build_feature_table([seq])

    taste = le_taste.inverse_transform(taste_model.predict(Xp))[0]
    sol = le_sol.inverse_transform(sol_model.predict(Xp))[0]
    dock = dock_model.predict(Xp)[0]

    st.markdown(f"""
    <div class="card">
        <div class="metric">Taste: {taste}</div>
        <div class="metric">Solubility: {sol}</div>
        <div class="metric">Docking score: {dock:.2f} kcal/mol</div>
    </div>
    """, unsafe_allow_html=True)

    feats = physicochemical_features(seq)
    comp = composition_features(seq)

    st.write("### Physicochemical properties")
    for k, v in feats.items():
        st.write(f"{k}: {v}")

    st.write("### Composition summary")
    for k, v in comp.items():
        st.write(f"{k}: {v}")

    st.session_state.pdb_text = build_peptide_pdb(seq)
    st.components.v1.html(show_structure(st.session_state.pdb_text)._make_html(), height=500)

# ==========================================================
# PDB UPLOAD + ANALYSIS
# ==========================================================

st.markdown("## 🧩 Upload & Analyze PDB")

uploaded_pdb = st.file_uploader("Upload PDB file", type=["pdb"], key="pdb_upload")

if uploaded_pdb:
    st.session_state.pdb_text = uploaded_pdb.read().decode()

if st.session_state.pdb_text:

    st.components.v1.html(show_structure(st.session_state.pdb_text)._make_html(), height=500)

    st.write("### Ramachandran plot")
    phi_psi = ramachandran_from_pdb(st.session_state.pdb_text)

    if len(phi_psi) == 0:
        st.warning("Ramachandran plot not available for this structure.")
    else:
        phi, psi = zip(*phi_psi)
        fig, ax = plt.subplots()
        ax.scatter(phi, psi, s=20)
        ax.set_xlim(-180, 180)
        ax.set_ylim(-180, 180)
        st.pyplot(fig)

    st.write("### Cα distance map")
    fig, ax = plt.subplots(figsize=(5,5))
    sns.heatmap(ca_distance_map(st.session_state.pdb_text), cmap="viridis", ax=ax)
    st.pyplot(fig)

# ==========================================================
# BATCH PREDICTION
# ==========================================================

st.markdown("## 📦 Batch Prediction")

batch_file = st.file_uploader("Upload CSV with 'peptide' column", type=["csv"], key="batch_upload")

if batch_file:
    batch_df = pd.read_csv(batch_file)

    if "peptide" not in batch_df.columns:
        st.error("CSV must contain a column named 'peptide'")
    else:
        batch_df["peptide"] = batch_df["peptide"].apply(clean_sequence)
        Xb = build_feature_table(batch_df["peptide"])

        batch_df["Predicted Taste"] = le_taste.inverse_transform(taste_model.predict(Xb))
        batch_df["Predicted Solubility"] = le_sol.inverse_transform(sol_model.predict(Xb))
        batch_df["Predicted Docking Score"] = dock_model.predict(Xb)

        st.dataframe(batch_df)
        st.download_button(
            "Download batch predictions",
            batch_df.to_csv(index=False),
            "batch_predictions.csv"
        )

# ==========================================================
# ANALYTICS
# ==========================================================

st.markdown("## 📊 Model Analytics")

for k, v in metrics.items():
    st.write(f"{k}: {round(v, 3)}")

coords = PCA(2).fit_transform(X_all)
fig, ax = plt.subplots()
ax.scatter(coords[:, 0], coords[:, 1])
st.pyplot(fig)

# ==========================================================
# FOOTER
# ==========================================================

st.markdown("""
<div class="footer">
© 2025 PepTastePredictor • Built for research & education
</div>
""", unsafe_allow_html=True)
