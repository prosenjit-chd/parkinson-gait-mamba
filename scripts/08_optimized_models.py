"""
scripts/08_optimized_models.py
--------------------------------------------------------
Deliverable 2: Baseline & Optimized Models (Corrected)
Deadline: Jan 11th, 2025 Requirements

Models (5 total):
1. RandomForest (Mandatory)
2. SVM - Linear Kernel (Mandatory)
3. SVM - RBF Kernel (Mandatory)
4. ExtraTrees (Selected for high AUC)
5. CatBoost (Selected for high Accuracy)

Methodology:
- Z-score normalization (StandardScaler)
- Stratified 5-Fold Cross-Validation
- Hyperparameter Optimization (RandomizedSearchCV)
    - SVM-RBF: Optimize 'C' and 'gamma'
    - SVM-Linear: Optimize 'C'
- Outputs: 
    - 15 Confusion Matrices (5 models * 3 inputs)
    - CSV with Accuracy, Sensitivity, Specificity, AUC
--------------------------------------------------------
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# ML Imports
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC

# Boosting
try:
    from catboost import CatBoostClassifier
except ImportError:
    print("[ERROR] Missing CatBoost. Installing automatically...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "catboost", "--quiet"])
    from catboost import CatBoostClassifier

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
TAB_DIR = BASE_DIR / "outputs" / "tables"
FIG_DIR = BASE_DIR / "outputs" / "figures" / "deliverable2"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Data Loading
# ------------------------------------------------------------
def load_data(foot_side):
    """Load TSFresh features for L, R, or LR."""
    feature_file = TAB_DIR / f"subject_features_{foot_side}.csv"
    demo_file = TAB_DIR / "demographics_clean.csv"

    if not feature_file.exists():
        raise FileNotFoundError(f"Feature file not found: {feature_file}")
    
    df_features = pd.read_csv(feature_file)
    df_demo = pd.read_csv(demo_file)
    label_map = {row['ID']: 1 if row['Group'] == 'PD' else 0 for _, row in df_demo.iterrows()}
    
    # ID cleaning
    if 'ID' not in df_features.columns:
        if 'subject' in df_features.columns:
            df_features.rename(columns={'subject': 'ID'}, inplace=True)
        else:
            raise ValueError("Feature file must have 'ID'")

    valid_ids = [pid for pid in df_features['ID'] if pid in label_map]
    df_features = df_features[df_features['ID'].isin(valid_ids)].copy()
    
    y = np.array([label_map[pid] for pid in df_features['ID']])
    X = df_features.drop(columns=['ID']).values
    
    return X, y

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def calc_specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0

def save_confusion_matrix(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    # Use larger font for visibility as requested
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Control', 'PD'], yticklabels=['Control', 'PD'],
                annot_kws={"size": 16, "weight": "bold"})
    plt.title(title, fontsize=14)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

# ------------------------------------------------------------
# Model Configs
# ------------------------------------------------------------
def get_model_config(model_type, random_state=42):
    
    # "The most common approach is to apply z-score scaling..." -> StandardScaler
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler() 
    selector = SelectKBest(f_classif) 
    
    # K-Best Range
    k_range = [20, 50, 100, "all"]
    
    config = {}
    
    if model_type == "RandomForest":
        model = RandomForestClassifier(random_state=random_state, n_jobs=1)
        params = {
            "selector__k": k_range,
            "clf__n_estimators": [100, 200, 300],
            "clf__max_depth": [None, 10, 20],
            "clf__min_samples_split": [2, 5, 10],
            "clf__class_weight": [None, "balanced"]
        }
    
    elif model_type == "SVM_Linear":
        model = SVC(kernel='linear', probability=True, random_state=random_state)
        params = {
            "selector__k": k_range,
            "clf__C": [0.01, 0.1, 1, 10, 100],
            "clf__class_weight": [None, "balanced"]
        }
        
    elif model_type == "SVM_RBF":
        model = SVC(kernel='rbf', probability=True, random_state=random_state)
        params = {
            "selector__k": k_range,
            "clf__C": [0.1, 1, 10, 100],
            "clf__gamma": ["scale", "auto", 0.01, 0.1],
            "clf__class_weight": [None, "balanced"]
        }
        
    elif model_type == "ExtraTrees":
        model = ExtraTreesClassifier(random_state=random_state, n_jobs=1)
        params = {
            "selector__k": k_range,
            "clf__n_estimators": [100, 200, 300],
            "clf__max_depth": [None, 10, 20],
            "clf__min_samples_split": [2, 5],
            "clf__class_weight": [None, "balanced"]
        }
        
    elif model_type == "CatBoost":
        model = CatBoostClassifier(verbose=False, random_state=random_state, allow_writing_files=False)
        params = {
            "selector__k": k_range,
            "clf__iterations": [100, 200],
            "clf__learning_rate": [0.01, 0.05, 0.1],
            "clf__depth": [4, 6, 8],
        }
        
    else:
        raise ValueError(f"Unknown model: {model_type}")

    pipeline = Pipeline([
        ('imputer', imputer),
        ('scaler', scaler),
        ('selector', selector),
        ('clf', model)
    ])
    return pipeline, params

# ------------------------------------------------------------
# Execution
# ------------------------------------------------------------
def main():
    # The 5 Required Models
    MODEL_LIST = ["RandomForest", "SVM_Linear", "SVM_RBF", "ExtraTrees", "CatBoost"]
    INPUTS = ["L", "R", "LR"]
    
    all_results = []
    
    print("\n[START] Starting Corrected Deliverable 2 Optimization (5 Models x 3 Inputs)")
    print(f"   Models: {MODEL_LIST}")
    
    for side in INPUTS:
        print(f"\n[LOAD] Loading Data: {side}")
        try:
            X, y = load_data(side)
        except Exception as e:
            print(f"Skipping {side}: {e}")
            continue
            
        outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for m_name in MODEL_LIST:
            print(f"   [TRAIN] Training {m_name} ({side})...")
            
            y_true_all = []
            y_pred_all = []
            y_prob_all = []
            
            # Stratified 5-Fold Cross Validation
            for train_idx, test_idx in outer_cv.split(X, y):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                pipe, param_grid = get_model_config(m_name)
                
                # RandomizedSearchCV (Optimize C, gamma, etc.)
                rs = RandomizedSearchCV(
                    pipe, param_distributions=param_grid, 
                    n_iter=10, cv=3, scoring='roc_auc', n_jobs=-1, random_state=42, verbose=0
                )
                rs.fit(X_train, y_train)
                best_model = rs.best_estimator_
                
                y_pred = best_model.predict(X_test)
                y_prob = best_model.predict_proba(X_test)[:, 1]
                
                y_true_all.extend(y_test)
                y_pred_all.extend(y_pred)
                y_prob_all.extend(y_prob)
                
            # Metrics
            acc = accuracy_score(y_true_all, y_pred_all)
            sens = recall_score(y_true_all, y_pred_all)
            spec = calc_specificity(y_true_all, y_pred_all)
            auc = roc_auc_score(y_true_all, y_prob_all)
            
            print(f"      [OK] Acc: {acc:.3f} | AUC: {auc:.3f}")
            
            # Save Result
            all_results.append({
                "Input": side,
                "Model": m_name,
                "Accuracy": acc,
                "Sensitivity": sens,
                "Specificity": spec,
                "AUC": auc
            })
            
            # Save Confusion Matrix
            cm_filename = FIG_DIR / f"CM_{side}_{m_name}.png"
            title = f"{m_name} ({side})\nAcc: {acc:.2f} AUC: {auc:.2f}"
            save_confusion_matrix(y_true_all, y_pred_all, title, cm_filename)
            
    # Save Final CSV
    if all_results:
        df = pd.DataFrame(all_results)
        out_csv = TAB_DIR / "results_deliverable2_final.csv"
        df.to_csv(out_csv, index=False)
        print(f"\n[SAVED] Saved Final Results to: {out_csv}")
        # Sort by AUC to show best performance
        print(df.sort_values(["Input", "AUC"], ascending=False).to_string())
    else:
        print("[ERROR] No results generated.")

if __name__ == "__main__":
    main()
