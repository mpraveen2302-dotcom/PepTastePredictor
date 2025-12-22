import streamlit as st
import pandas as pd
from features import clean_sequence, extract_features
from models import train_models
from structure import build_ca_pdb, show_structure
from visualization import plot_confusion, plot_pca, plot_regression
from batch import batch_predict
from report import generate_report

st.set_page_config(page_title="PepTastePredictor", layout="wide")

DATASET_PATH = "AIML (4).xlsx"
df = pd.read_excel(DATASET_PATH)
df.columns = df.columns.str.lower().str.strip()

(staste, ssol, sdock,
 le_taste, le_sol,
 metrics, X, yt, ys, yd) = train_models(df)

st.title("🧬 PepTastePredictor")

st.header("Single Peptide Prediction")
seq = st.text_input("Enter peptide sequence")

if st.button("Predict"):
    seq = clean_sequence(seq)
    Xp = pd.DataFrame(extract_features([seq]))
    taste = le_taste.inverse_transform(staste.predict(Xp))[0]
    sol = le_sol.inverse_transform(ssol.predict(Xp))[0]
    dock = sdock.predict(Xp)[0]

    st.success(f"""
    **Taste:** {taste}  
    **Solubility:** {sol}  
    **Docking Score:** {dock:.2f}
    """)

    pdb = build_ca_pdb(seq)
    st.download_button("Download PDB", pdb, file_name="peptide.pdb")
    show_structure(pdb)

st.header("Batch Prediction")
file = st.file_uploader("Upload CSV with peptide column")
if file:
    out = batch_predict(file, staste, ssol, sdock, le_taste, le_sol)
    st.dataframe(out)
    st.download_button("Download Results", out.to_csv(index=False), "batch_results.csv")

st.header("Model Analytics")
st.metric("Taste Accuracy", metrics["Taste Accuracy"])
st.metric("Docking RMSE", metrics["Docking RMSE"])

st.pyplot(plot_confusion(yt, staste.predict(X), le_taste.classes_, "Taste Confusion"))
st.pyplot(plot_pca(X, yt))
st.pyplot(plot_regression(yd, sdock.predict(X)))

if st.button("Download PDF Report"):
    path = generate_report(metrics)
    st.download_button("Download Report", open(path,"rb"), file_name=path)
