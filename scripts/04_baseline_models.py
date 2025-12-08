import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


# ====================================================
#  MAIN FUNCTION FOR ANY FEATURE SET
# ====================================================
def evaluate_models(X, y, feature_name="LEFT"):
    print("\n" + "=" * 60)
    print(f"BASELINE RESULTS FOR: {feature_name}")
    print("=" * 60)

    # 1) Stratified 5-fold CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 2) Standardization
    scaler = StandardScaler()

    # 3) Classifiers + Hyperparameters
    models = {
        "RandomForest": (
            RandomForestClassifier(),
            {"n_estimators": [100, 300], "max_depth": [5, 10, None]},
        ),
        "SVM_Linear": (
            SVC(kernel="linear", probability=True),
            {"C": [0.1, 1, 10, 100]},
        ),
        "SVM_RBF": (
            SVC(kernel="rbf", probability=True),
            {"C": [0.1, 1, 10, 100], "gamma": ["scale", 0.01, 0.001]},
        ),
    }

    best_results = {}

    # ================================================
    #  4) TRAINING + GRID SEARCH + TESTING
    # ================================================
    for model_name, (clf, params) in models.items():
        print(f"\n🔹 Training model: {model_name}")

        grid = GridSearchCV(
            estimator=clf,
            param_grid=params,
            cv=skf,
            scoring="accuracy",
            n_jobs=-1,
        )

        # Fit with scaled data
        X_scaled = scaler.fit_transform(X)
        grid.fit(X_scaled, y)

        best_clf = grid.best_estimator_

        # Predictions for full dataset (you can also separate test set if you want)
        y_pred = best_clf.predict(X_scaled)
        y_prob = best_clf.predict_proba(X_scaled)[:, 1]

        # Metrics
        acc = accuracy_score(y, y_pred)
        sensitivity = recall_score(y, y_pred, pos_label=1)
        specificity = recall_score(y, y_pred, pos_label=0)
        auc = roc_auc_score(y, y_prob)
        cm = confusion_matrix(y, y_pred)

        results = {
            "accuracy": acc,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "auc": auc,
            "confusion_matrix": cm,
            "best_params": grid.best_params_,
        }

        best_results[model_name] = results

        print(f"  ➤ Best params: {grid.best_params_}")
        print(f"  ➤ Accuracy: {acc:.4f}")
        print(f"  ➤ Sensitivity: {sensitivity:.4f}")
        print(f"  ➤ Specificity: {specificity:.4f}")
        print(f"  ➤ AUC: {auc:.4f}")
        print(f"  ➤ Confusion Matrix:\n{cm}")

    return best_results


# ====================================================
#  MAIN SCRIPT (REPLACE THESE WITH YOUR ACTUAL FEATURE MATRICES)
# ====================================================
if __name__ == "__main__":
    # Load your features saved in Deliverable-1
    # Example placeholders:
    X_left = np.load("X_left.npy")  # tsfresh left
    X_right = np.load("X_right.npy")  # tsfresh right
    X_combined = np.load("X_combined.npy")  # concatenated features
    y = np.load("y.npy")  # labels

    # Run experiments
    left_results = evaluate_models(X_left, y, feature_name="LEFT FOOT")
    right_results = evaluate_models(X_right, y, feature_name="RIGHT FOOT")
    comb_results = evaluate_models(X_combined, y, feature_name="COMBINED")

    # Save results
    pd.to_pickle(left_results, "baseline_left.pkl")
    pd.to_pickle(right_results, "baseline_right.pkl")
    pd.to_pickle(comb_results, "baseline_combined.pkl")

    print("\n🎉 All baseline model experiments completed!")
