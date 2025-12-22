# 🧬 PepTastePredictor

**PepTastePredictor** is an advanced **Streamlit-based peptide analysis and prediction platform** that integrates **bioinformatics, machine learning, and structural biology** into a single unified application.

The app predicts:
- **Taste profile**
- **Solubility**
- **Docking score (kcal/mol)**

and provides:
- **Extensive feature extraction (400+ features)**
- **Model performance analytics**
- **Multiple visualizations**
- **3D peptide structure (PDB) generation**
- **Direct AlphaFold / ColabFold integration**

This platform is suitable for **research, education, and AI-driven peptide design**.

---

## 🚀 Key Features

### 🔬 1. Extensive Feature Engineering (400+ Features)

PepTastePredictor extracts a **large and diverse feature set** from peptide sequences:

#### Physicochemical Properties
- Peptide length  
- Molecular weight  
- Isoelectric point (pI)  
- Net charge at pH 7  
- GRAVY (hydrophobicity index)  
- Aromaticity  
- Instability index  

#### Structural Propensity (BioPython-based)
- Alpha-helix fraction  
- Beta-turn fraction  
- Beta-sheet fraction  

#### Sequence-Based Features
- Amino acid composition (20 features)
- Dipeptide composition (400 features)

➡️ **Total features per peptide: 400+**

---

## 🤖 2. Machine Learning Models

The app uses **Random Forest models**:

| Task | Model |
|-----|------|
| Taste prediction | RandomForestClassifier |
| Solubility prediction | RandomForestClassifier |
| Docking score prediction | RandomForestRegressor |

### Training Strategy
- Automatic **train–test split**
- Stratified splitting when possible
- Robust handling of small or imbalanced datasets

---

## 📊 3. Model Performance Metrics

PepTastePredictor reports **clear quantitative evaluation**:

### Classification
- Accuracy
- F1-score (weighted)

### Regression
- RMSE (Root Mean Squared Error)
- R² score

These metrics are printed in the console and used internally for validation.

---

## 📈 4. Visualization & Analysis

The app provides **independent, correctly rendered visualizations**:

### Confusion Matrices
- Taste classification confusion matrix
- Solubility classification confusion matrix

### Feature Space Analysis
- PCA (2D) plot for taste-based clustering

### Regression Quality
- Actual vs Predicted Docking Score plot

Each plot is generated using **fresh figures** (no reused global plots).

---

## 🧬 5. Peptide Structure & AlphaFold Integration

### PDB Generation
- Automatically generates a **CA-trace PDB file**
- File is downloadable directly from the app

### AlphaFold / ColabFold
- Direct link provided to run **structure prediction**
- Compatible with generated peptide sequences and uploaded PDBs

🔗 AlphaFold Notebook  
https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/AlphaFold2_mmseqs2_advanced.ipynb

---

## 🧪 6. Streamlit User Interface

### Single Unified App
- One-page Streamlit app
- No multiple sub-apps or split interfaces

### Inputs
- Peptide sequence (text input)

### Outputs
- Predicted taste
- Predicted solubility
- Predicted docking score
- Total feature count
- Downloadable PDB file
- Analytical plots

---

## 📁 Dataset Requirements

Your dataset must contain the following columns (case-insensitive):

