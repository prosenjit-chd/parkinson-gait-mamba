"""
07_baseline_figures.py

Generates all Task-2 figures from your baseline models:
- Bar plots: Accuracy, Sensitivity, Specificity, AUC
- Metric heatmap
- ROC curves per foot (SVM-linear, SVM-RBF, RandomForest)
- Confusion matrices for best model per foot (by AUC)

Assumptions:
- baseline_results.json already exists in outputs/tables/
- subject_features_L.csv, subject_features_R.csv, subject_features_LR.csv
  and demographics_clean.csv are in outputs/tables/
- Same preprocessing as 06_baseline_models.py (SimpleImputer + StandardScaler)
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    recall_score,
)


# --------------------------------------------------------------------
# PATHS
# --------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
TABLE_DIR = BASE_DIR / "outputs" / "tables"
FIG_DIR = BASE_DIR / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

BASELINE_JSON = TABLE_DIR / "baseline_results.json"

FILE_L = TABLE_DIR / "subject_features_L.csv"
FILE_R = TABLE_DIR / "subject_features_R.csv"
FILE_LR = TABLE_DIR / "subject_features_LR.csv"
DEMO_FILE = TABLE_DIR / "demographics_clean.csv"


# --------------------------------------------------------------------
# LOAD BASELINE RESULTS (for bar plots + heatmap)
# --------------------------------------------------------------------
def load_baseline_results():
    with open(BASELINE_JSON, "r") as f:
        results = json.load(f)

    rows = []
    for foot, models in results.items():
        for model_name, metrics in models.items():
            rows.append(
                {
                    "Foot": foot,
                    "Model": model_name,
                    "Accuracy": metrics["Accuracy"],
                    "Sensitivity": metrics["Sensitivity"],
                    "Specificity": metrics["Specificity"],
                    "AUC": metrics["AUC"],
                }
            )
    df = pd.DataFrame(rows)
    return df


# --------------------------------------------------------------------
# BAR PLOTS + HEATMAP
# --------------------------------------------------------------------
def plot_metric_bars(df, metric_name, ylabel):
    """
    Creates bar plot for given metric using baseline_results.json
    x-axis = Foot/Model combination
    """
    plt.figure(figsize=(8, 5))
    labels = []
    values = []

    # order for consistent plotting
    foot_order = ["LEFT", "RIGHT", "COMBINED"]
    model_order = ["SVM_linear", "SVM_rbf", "RandomForest"]

    for foot in foot_order:
        for model in model_order:
            row = df[(df["Foot"] == foot) & (df["Model"] == model)].iloc[0]
            labels.append(f"{foot}\n{model}")
            values.append(row[metric_name])

    x = np.arange(len(labels))
    plt.bar(x, values)
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel(ylabel)
    plt.title(f"{metric_name} comparison (5-fold CV mean)")
    plt.tight_layout()

    out_path = FIG_DIR / f"{metric_name.lower()}_comparison.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved: {out_path}")


def plot_metrics_heatmap(df):
    """
    Heatmap-like overview for all metrics (Foot x Model grid)
    """
    metrics = ["Accuracy", "Sensitivity", "Specificity", "AUC"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    for ax, metric in zip(axes.ravel(), metrics):
        pivot = df.pivot(index="Foot", columns="Model", values=metric)
        im = ax.imshow(pivot.values, aspect="auto")

        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_title(metric)

        # annotate with values
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                ax.text(
                    j,
                    i,
                    f"{pivot.values[i, j]:.2f}",
                    ha="center",
                    va="center",
                )

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    out_path = FIG_DIR / "all_metrics_heatmap.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved: {out_path}")


# --------------------------------------------------------------------
# LOAD FEATURES + LABELS (for ROC + confusion matrices)
# --------------------------------------------------------------------
def load_features_and_labels():
    df_L = pd.read_csv(FILE_L)
    df_R = pd.read_csv(FILE_R)
    df_LR = pd.read_csv(FILE_LR)
    df_demo = pd.read_csv(DEMO_FILE)

    # CO=0, PD=1
    y = np.array([1 if g == "PD" else 0 for g in df_demo["Group"].values])

    X_L = df_L.drop(columns=["ID"]).values
    X_R = df_R.drop(columns=["ID"]).values
    X_LR = df_LR.drop(columns=["ID"]).values

    return {"LEFT": X_L, "RIGHT": X_R, "COMBINED": X_LR}, y


# --------------------------------------------------------------------
# RE-TRAIN MODELS FOR ROC + CONFUSION
# --------------------------------------------------------------------
def get_model_definitions():
    models = {
        "SVM_linear": (
            SVC(kernel="linear", probability=True),
            {"C": [0.1, 1, 10]},
        ),
        "SVM_rbf": (
            SVC(kernel="rbf", probability=True),
            {"C": [0.1, 1, 10], "gamma": ["scale"]},
        ),
        "RandomForest": (
            RandomForestClassifier(),
            {"n_estimators": [200], "max_depth": [None, 20]},
        ),
    }
    return models


def crossval_predictions(X, y, model, param_grid, n_splits=5, random_state=42):
    """
    Run StratifiedKFold + GridSearchCV,
    return concatenated y_true, y_pred, y_proba for plotting.
    """

    # Impute NaNs once globally
    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(X)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    all_y_true = []
    all_y_pred = []
    all_y_proba = []

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        clf = GridSearchCV(
            model,
            param_grid,
            cv=3,
            scoring="accuracy",
            n_jobs=-1,
        )
        clf.fit(X_train_scaled, y_train)
        best = clf.best_estimator_

        proba = best.predict_proba(X_val_scaled)[:, 1]
        preds = best.predict(X_val_scaled)

        all_y_true.append(y_val)
        all_y_pred.append(preds)
        all_y_proba.append(proba)

    all_y_true = np.concatenate(all_y_true)
    all_y_pred = np.concatenate(all_y_pred)
    all_y_proba = np.concatenate(all_y_proba)

    return all_y_true, all_y_pred, all_y_proba


# --------------------------------------------------------------------
# ROC CURVES + CONFUSION MATRICES
# --------------------------------------------------------------------
def plot_roc_curves_per_foot(X_dict, y):
    """
    For each foot (LEFT, RIGHT, COMBINED):
      - ROC curves for SVM_linear, SVM_rbf, RandomForest
      - Confusion matrix for best model (by AUC)
    """
    models = get_model_definitions()

    for foot_name, X in X_dict.items():
        print(f"\n=== ROC / Confusion for {foot_name} ===")

        plt.figure(figsize=(6, 5))

        model_results = {}

        # run all models, collect ROC + confusion info
        for model_name, (model, param_grid) in models.items():
            y_true, y_pred, y_proba = crossval_predictions(X, y, model, param_grid)

            fpr, tpr, _ = roc_curve(y_true, y_proba)
            auc_val = roc_auc_score(y_true, y_proba)

            model_results[model_name] = {
                "y_true": y_true,
                "y_pred": y_pred,
                "y_proba": y_proba,
                "fpr": fpr,
                "tpr": tpr,
                "auc": auc_val,
            }

            plt.plot(fpr, tpr, label=f"{model_name} (AUC={auc_val:.3f})")

        # diagonal line
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curves – {foot_name}")
        plt.legend()
        plt.tight_layout()

        roc_path = FIG_DIR / f"roc_{foot_name.lower()}.png"
        plt.savefig(roc_path, dpi=200)
        plt.close()
        print(f"Saved: {roc_path}")

        # Choose best model by AUC for confusion matrix
        best_model_name = max(
            model_results.keys(),
            key=lambda m: model_results[m]["auc"],
        )
        print(f"Best model for {foot_name}: {best_model_name}")

        best_res = model_results[best_model_name]
        cm = confusion_matrix(best_res["y_true"], best_res["y_pred"])
        tn, fp, fn, tp = cm.ravel()
        acc = accuracy_score(best_res["y_true"], best_res["y_pred"])
        sens = recall_score(best_res["y_true"], best_res["y_pred"])
        spec = tn / (tn + fp)

        # Plot confusion matrix
        plt.figure(figsize=(4, 4))
        plt.imshow(cm, interpolation="nearest")
        plt.title(f"{foot_name} – {best_model_name}\nConfusion Matrix")
        plt.colorbar()
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, ["Control (0)", "PD (1)"])
        plt.yticks(tick_marks, ["Control (0)", "PD (1)"])

        # add text
        for i in range(2):
            for j in range(2):
                plt.text(
                    j,
                    i,
                    str(cm[i, j]),
                    ha="center",
                    va="center",
                )

        plt.xlabel(f"Predicted\nAcc={acc:.2f}, Sens={sens:.2f}, Spec={spec:.2f}")
        plt.ylabel("True Label")
        plt.tight_layout()

        cm_path = FIG_DIR / f"confusion_{foot_name.lower()}_{best_model_name}.png"
        plt.savefig(cm_path, dpi=200)
        plt.close()
        print(f"Saved: {cm_path}")


# --------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------
if __name__ == "__main__":
    print("🔍 Loading baseline_results.json for bar plots & heatmap...")
    df_baseline = load_baseline_results()

    print("📊 Creating bar plots...")
    plot_metric_bars(df_baseline, "Accuracy", "Accuracy")
    plot_metric_bars(df_baseline, "Sensitivity", "Sensitivity")
    plot_metric_bars(df_baseline, "Specificity", "Specificity")
    plot_metric_bars(df_baseline, "AUC", "AUC")

    print("📊 Creating metrics heatmap...")
    plot_metrics_heatmap(df_baseline)

    print("\n🔍 Loading features & labels for ROC / confusion matrices...")
    X_dict, y = load_features_and_labels()

    print("📉 Creating ROC curves + confusion matrices...")
    plot_roc_curves_per_foot(X_dict, y)

    print("\n✅ All Task-2 figures generated in:")
    print(FIG_DIR)
