import pandas as pd
from features import extract_features

def batch_predict(file, taste_model, sol_model, dock_model, le_taste, le_sol):
    df = pd.read_csv(file)
    X = pd.DataFrame(extract_features(df["peptide"]))
    df["Predicted Taste"] = le_taste.inverse_transform(taste_model.predict(X))
    df["Predicted Solubility"] = le_sol.inverse_transform(sol_model.predict(X))
    df["Predicted Docking"] = dock_model.predict(X)
    return df
