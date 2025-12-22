import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from features import extract_features

def train_models(df):
    X = pd.DataFrame(extract_features(df["peptide"]))
    le_taste = LabelEncoder()
    le_sol = LabelEncoder()

    y_taste = le_taste.fit_transform(df["taste"])
    y_sol = le_sol.fit_transform(df["solubility"])
    y_dock = df["docking score (kcal/mol)"].astype(float)

    X_tr, X_te, yt_tr, yt_te, ys_tr, ys_te, yd_tr, yd_te = train_test_split(
        X, y_taste, y_sol, y_dock, test_size=0.2, random_state=42
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

    return taste_model, sol_model, dock_model, le_taste, le_sol, metrics, X, y_taste, y_sol, y_dock
