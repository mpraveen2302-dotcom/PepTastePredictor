import py3Dmol
import streamlit as st

def build_ca_pdb(seq):
    x = 0.0
    lines = []
    for i, aa in enumerate(seq, 1):
        lines.append(
            f"ATOM  {i:5d}  CA  {aa} A{i:4d}    {x:8.3f} 0.000 0.000  1.00  0.00           C"
        )
        x += 3.8
    return "\n".join(lines) + "\nEND"

def show_structure(pdb_text):
    view = py3Dmol.view(width=600, height=500)
    view.addModel(pdb_text, "pdb")
    view.setStyle({"cartoon": {"color": "spectrum"}})
    view.zoomTo()
    st.components.v1.html(view._make_html(), height=500)
