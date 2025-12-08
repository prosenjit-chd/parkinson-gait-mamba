"""
05_baseline_models_final.py
--------------------------------------------------------
Implements Deliverable #2: Baseline Models
- Random Forest
- SVM (Linear)
- SVM (RBF)

Methodology:
- Stratified 5-Fold Cross-Validation
- Grid Search for Hyperparameter Tuning (nested within CV)
- Z-score normalization (StandardScaler) inside the pipeline
- Metrics: Accuracy, Sensitivity, Specificity, AUC
- Plots: Confusion Matrices for the best models

Inputs:
- outputs/tables/features_L.csv
- outputs/tables/features_R.csv
- outputs/tables/features_LR.csv

Outputs:
- outputs/tables/results_baseline.csv
- outputs/figures/confusion_matrices/
--------------------------------------------------------
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend("Agg")
import seaborn as sns
from pathlib import Path
from tqdm import tqdm

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    confusion_matrix,
    roc_auc_score,
    make_scorer
)

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
TAB_DIR = Path("outputs/tables")
FIG_DIR = Path("outputs/figures/confusion_matrices")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Define models and hyperparameter grids
MODELS = {
    "RandomForest": {
        "model": RandomForestClassifier(random_state=42),
        "params": {
            "clf__n_estimators": [50, 100, 200],
            "clf__max_depth": [None, 10, 20],
            "clf__min_samples_split": [2, 5]
        }
    },
    "SVM_Linear": {
        "model": SVC(kernel="linear", probability=True, random_state=42),
        "params": {
            "clf__C": [0.01, 0.1, 1, 10, 100]
        }
    },
    "SVM_RBF": {
        "model": SVC(kernel="rbf", probability=True, random_state=42),
        "params": {
            "clf__C": [0.1, 1, 10, 100],
            "clf__gamma": ["scale", "auto", 0.01, 0.1]
        }
    }
}

# ------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------

def calculate_specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)

def plot_confusion_matrix(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Control", "PD"], yticklabels=["Control", "PD"])
    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def run_experiment(feature_file, tag):
    print(f"\n🚀 Processing: {tag} ({feature_file})")
    
    # Load Data
    if not feature_file.exists():
        print(f"❌ File not found: {feature_file}")
        return []

    df = pd.read_csv(feature_file)
    
    # Prepare X and y
    # Assuming 'subject' is the index or column, and labels need to be derived or are present
    # The previous script saved 'subject' as a column. We need labels.
    # We'll re-load labels based on subject ID naming convention or metadata.
    # For now, let's assume the previous script DID NOT save labels in the CSV explicitly,
    # but we can infer them from the subject ID (Control vs PD).
    # Actually, looking at 03_tsfresh... it saves 'subject' column. 
    # Let's infer label from subject string (e.g., "GaCo..." vs "GaPt...").
    
    # Filter out non-numeric columns for X
    X_cols = [c for c in df.columns if c not in ["subject", "label"]]
    X = df[X_cols].values
    
    # Generate labels
    # Convention: *Co* = Control (0), *Pt* = Patient (1)
    # This covers GaCo/GaPt, JuCo/JuPt, SiCo/SiPt
    subjects = df["subject"].astype(str)
    y = subjects.apply(lambda x: 0 if "Co" in x else 1).values
    
    print(f"   Data Shape: {X.shape} | Class Balance: {np.bincount(y)} (0=Control, 1=PD)")

    results = []
    
    # Outer CV Loop
    cv_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for model_name, config in MODELS.items():
        print(f"   👉 Training {model_name}...")
        
        fold_metrics = {"acc": [], "sens": [], "spec": [], "auc": []}
        all_y_true = []
        all_y_pred = []
        
        # We need to collect predictions for the "best" confusion matrix.
        # Since we do 5-fold, we can aggregate all validation predictions to make one big CM.
        
        for train_idx, test_idx in tqdm(cv_outer.split(X, y), total=5, leave=False):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Pipeline: Scale -> GridSearch(Model)
            # Note: We put GridSearch INSIDE the loop, but we can also put the Pipeline INSIDE GridSearch.
            # Best practice: GridSearch manages the validation split for tuning.
            # We use the pipeline as the estimator for GridSearch to ensure scaling happens correctly in inner loops.
            
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", config["model"])
            ])
            
            # Grid Search (Inner Loop)
            # We need to prefix params with 'clf__' because it's in a pipeline
            gs = GridSearchCV(
                pipe, 
                config["params"], 
                cv=3, # 3-fold for inner tuning to save time
                scoring="accuracy", 
                n_jobs=-1
            )
            
            gs.fit(X_train, y_train)
            best_model = gs.best_estimator_
            
            # Predict
            y_pred = best_model.predict(X_test)
            y_prob = best_model.predict_proba(X_test)[:, 1]
            
            # Metrics
            fold_metrics["acc"].append(accuracy_score(y_test, y_pred))
            fold_metrics["sens"].append(recall_score(y_test, y_pred)) # Sensitivity = Recall
            fold_metrics["spec"].append(calculate_specificity(y_test, y_pred))
            fold_metrics["auc"].append(roc_auc_score(y_test, y_prob))
            
            all_y_true.extend(y_test)
            all_y_pred.extend(y_pred)
            
        # Average metrics across folds
        avg_res = {
            "Model": model_name,
            "Input": tag,
            "Accuracy": np.mean(fold_metrics["acc"]),
            "Sensitivity": np.mean(fold_metrics["sens"]),
            "Specificity": np.mean(fold_metrics["spec"]),
            "AUC": np.mean(fold_metrics["auc"])
        }
        results.append(avg_res)
        
        # Plot Aggregate Confusion Matrix
        cm_title = f"{model_name} - {tag} (Acc: {avg_res['Accuracy']:.2f})"
        cm_path = FIG_DIR / f"cm_{model_name}_{tag}.png"
        plot_confusion_matrix(all_y_true, all_y_pred, cm_title, cm_path)
        
    return results

# ------------------------------------------------------------
# Main Execution
# ------------------------------------------------------------
def main():
    all_results = []
    
    # Process each feature set
    for tag in ["L", "R", "LR"]:
        fpath = TAB_DIR / f"features_{tag}.csv"
        res = run_experiment(fpath, tag)
        all_results.extend(res)
        
    # Save Results
    if all_results:
        df_res = pd.DataFrame(all_results)
        out_path = TAB_DIR / "results_baseline.csv"
        df_res.to_csv(out_path, index=False)
        print(f"\n✅ Results saved to {out_path}")
        print(df_res.round(3))
    else:
        print("\n❌ No results generated.")

if __name__ == "__main__":
    main()
