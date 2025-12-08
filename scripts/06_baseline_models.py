"""
Task-2: Baseline Models for PD vs Control
Handles NaN values using SimpleImputer
"""

import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    recall_score,
)
from sklearn.impute import SimpleImputer


# -------------------------------------------------------------
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TABLE_DIR = os.path.join(BASE, "outputs", "tables")

FILE_L = os.path.join(TABLE_DIR, "subject_features_L.csv")
FILE_R = os.path.join(TABLE_DIR, "subject_features_R.csv")
FILE_LR = os.path.join(TABLE_DIR, "subject_features_LR.csv")
DEMO_FILE = os.path.join(TABLE_DIR, "demographics_clean.csv")


# -------------------------------------------------------------
def load_data():
    df_L = pd.read_csv(FILE_L)
    df_R = pd.read_csv(FILE_R)
    df_LR = pd.read_csv(FILE_LR)
    df_demo = pd.read_csv(DEMO_FILE)

    y = np.array([1 if g == "PD" else 0 for g in df_demo["Group"].values])

    X_L = df_L.drop(columns=["ID"]).values
    X_R = df_R.drop(columns=["ID"]).values
    X_LR = df_LR.drop(columns=["ID"]).values

    return X_L, X_R, X_LR, y


# -------------------------------------------------------------
def train_models(X, y):
    imputer = SimpleImputer(strategy="median")  # 🔥 handles NaN safely
    X = imputer.fit_transform(X)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    models = {
        "SVM_linear": (SVC(kernel="linear", probability=True), {"C": [0.1, 1, 10]}),
        "SVM_rbf": (
            SVC(kernel="rbf", probability=True),
            {"C": [0.1, 1, 10], "gamma": ["scale", 0.01]},
        ),
        "RandomForest": (
            RandomForestClassifier(),
            {"n_estimators": [200], "max_depth": [None, 20]},
        ),
    }

    results = {}

    for name, (model, params) in models.items():
        fold_acc, fold_sens, fold_spec, fold_auc = [], [], [], []

        for train_idx, val_idx in skf.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)

            clf = GridSearchCV(model, params, cv=3, scoring="accuracy", n_jobs=-1)
            clf.fit(X_train, y_train)

            best = clf.best_estimator_
            preds = best.predict(X_val)
            probs = best.predict_proba(X_val)[:, 1]

            acc = accuracy_score(y_val, preds)
            sens = recall_score(y_val, preds)
            tn, fp, fn, tp = confusion_matrix(y_val, preds).ravel()
            spec = tn / (tn + fp)
            auc = roc_auc_score(y_val, probs)

            fold_acc.append(acc)
            fold_sens.append(sens)
            fold_spec.append(spec)
            fold_auc.append(auc)

        results[name] = {
            "Accuracy": float(np.mean(fold_acc)),
            "Sensitivity": float(np.mean(fold_sens)),
            "Specificity": float(np.mean(fold_spec)),
            "AUC": float(np.mean(fold_auc)),
            "Best Params": clf.best_params_,
        }

        print(f"\n===== {name} =====")
        print(results[name])

    return results


# -------------------------------------------------------------
if __name__ == "__main__":
    print("🔍 Loading subject-level TSFresh features...")
    X_L, X_R, X_LR, y = load_data()

    print("\n🚀 LEFT foot...")
    res_L = train_models(X_L, y)

    print("\n🚀 RIGHT foot...")
    res_R = train_models(X_R, y)

    print("\n🚀 COMBINED...")
    res_LR = train_models(X_LR, y)

    import json

    out_file = os.path.join(TABLE_DIR, "baseline_results.json")

    with open(out_file, "w") as f:
        json.dump({"LEFT": res_L, "RIGHT": res_R, "COMBINED": res_LR}, f, indent=4)

    print(f"\n✅ Saved: {out_file}")
