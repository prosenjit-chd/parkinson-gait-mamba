"""
04_visualize_summary.py
Consolidated script that generates visual summaries side-by-side:
1. Demographics & Features Variance/Correlation Summary (from 04_visualize_csv_summary.py)
2. Feature-UPDRS Correlation Analysis (from compute_updrs_correlation.py and updrsFeatureCorrelationVisualization.py)
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

TABLE_DIR = Path("outputs/tables")
FIG_DIR = Path("outputs/figures")

FIG_DIR.mkdir(exist_ok=True, parents=True)

print(" Creating Demographics and Features Variance Summaries...")

# ------------------------------------------------------------
# 1. DEMOGRAPHICS SUMMARY PLOT
# ------------------------------------------------------------
try:
    df_demo = pd.read_csv(TABLE_DIR / "demographics_summary.csv")
    # For compatibility with older demographic structure where Label might not be present
    group_col = "Label" if "Label" in df_demo.columns else "Group"
    count_col = "N" if "N" in df_demo.columns else "Subjects"

    plt.figure(figsize=(6, 4))
    sns.barplot(x=group_col, y=count_col, data=df_demo, palette="Blues_d")
    plt.title("Number of Subjects per Group")
    plt.xlabel("Group")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "subjects_per_group.png", dpi=150)
    plt.close()
    print(" Created subjects_per_group.png")
except Exception as e:
    print(f"Skipping demographic plot: {e}")

# ------------------------------------------------------------
# 2. FEATURE VARIANCE DISTRIBUTION (for LR combined)
# ------------------------------------------------------------
try:
    df_feat = pd.read_csv(TABLE_DIR / "features_LR.csv", index_col=0)
    var_series = df_feat.var().sort_values(ascending=False)
    top10 = var_series.head(10)

    plt.figure(figsize=(7, 4))
    sns.barplot(x=top10.values, y=top10.index, palette="magma")
    plt.title("Top 10 Features by Variance (Combined Feet)")
    plt.xlabel("Variance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "top_features_variance_LR.png", dpi=150)
    plt.close()
    print(" Created top_features_variance_LR.png")

    # ------------------------------------------------------------
    # 3. FEATURE CORRELATION HEATMAP (for LR combined)
    # ------------------------------------------------------------
    corr = df_feat.corr().abs()
    plt.figure(figsize=(6, 5))
    sns.heatmap(corr.iloc[:10, :10], cmap="coolwarm", annot=False)
    plt.title("Feature Correlation Heatmap (First 10 Features)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "feature_corr_heatmap_LR.png", dpi=150)
    plt.close()
    print(" Created feature_corr_heatmap_LR.png")
except Exception as e:
    print(f"Skipping feature variance/correlation plots: {e}")

try:
    # ------------------------------------------------------------
    # 4. COMPARISON OF LEFT VS RIGHT FEATURE MEANS
    # ------------------------------------------------------------
    df_L = pd.read_csv(TABLE_DIR / "features_L.csv", index_col=0)
    df_R = pd.read_csv(TABLE_DIR / "features_R.csv", index_col=0)

    plt.figure(figsize=(6, 4))
    sns.kdeplot(df_L.mean(axis=1), label="Left Foot", color="blue")
    sns.kdeplot(df_R.mean(axis=1), label="Right Foot", color="orange")
    plt.title("Feature Mean Distribution per Subject")
    plt.xlabel("Mean Feature Value")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "feature_mean_distribution_LR.png", dpi=150)
    plt.close()
    print(" Created feature_mean_distribution_LR.png")
except Exception as e:
    print(f"Skipping left/right comparison: {e}")

# ------------------------------------------------------------
# 5. COMPUTE UPDRS CORRELATION AND PLOTS
# ------------------------------------------------------------
print(" Computing Feature-UPDRS Correlations...")
try:
    updrs_path = "data/metadata/demographics.xls"
    if not Path(updrs_path).exists():
        updrs_path = "data/metadata/demographics.xlsx" # Fallback

    updrs_df = pd.read_excel(updrs_path)
    
    # Process features_df again
    df_feat_full = pd.read_csv(TABLE_DIR / "features_LR.csv")
    
    # Extract clean IDs
    df_feat_full["subject_clean"] = df_feat_full["subject"].str.extract(r"([A-Za-z]+\d+)")
    updrs_df["ID_clean"] = updrs_df["ID"].astype(str).str.strip()
    updrs_df["UPDRS"] = pd.to_numeric(updrs_df["UPDRS"], errors="coerce")
    
    merged = pd.merge(
        df_feat_full, updrs_df, left_on="subject_clean", right_on="ID_clean", how="inner"
    )
    merged = merged.dropna(subset=["UPDRS"])

    corr_list = []
    for feature in df_feat_full.columns:
        if feature in ["subject", "subject_clean"]:
            continue
        if merged[feature].nunique() > 1:
            corr_val, _ = pearsonr(merged[feature], merged["UPDRS"])
            corr_list.append((feature, corr_val))
            
    df_corr = pd.DataFrame(corr_list, columns=["feature_name", "correlation"])
    df_corr["abs_corr"] = df_corr["correlation"].abs()
    df_corr = df_corr.sort_values(by="abs_corr", ascending=False)
    
    output_corr_csv = TABLE_DIR / "feature_updrs_correlation.csv"
    df_corr.to_csv(output_corr_csv, index=False)
    print(f" Saved Feature-UPDRS correlation table to: {output_corr_csv}")

    top_corr = df_corr.head(10)
    sns.set(style="whitegrid", font_scale=1.1)
    plt.rcParams["axes.labelsize"] = 11
    plt.rcParams["axes.titlesize"] = 13
    
    # Plot 1: Combined UPDRS
    plt.figure(figsize=(8, 5))
    sns.barplot(x="correlation", y="feature_name", data=top_corr, palette="viridis")
    plt.title("Top 10 TSFresh Features Correlated with UPDRS Score", fontsize=13, weight="bold")
    plt.xlabel("Correlation Coefficient (r)")
    plt.ylabel("Feature Name")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "updrsFeatureCorrelationVisualization.png", dpi=300)
    plt.close()
    
    # Plot 2: Signed
    plt.figure(figsize=(8, 5))
    sns.barplot(data=top_corr, x="correlation", y="feature_name", palette="coolwarm", hue="correlation", dodge=False, legend=False)
    plt.axvline(0, color="gray", linestyle="--", lw=1)
    plt.title("Top 10 TSFresh Features Correlated with UPDRS Score (Signed)", weight="bold")
    plt.xlabel("Correlation Coefficient (r)")
    plt.ylabel("Feature Name")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "UPDRS_Correlation_Signed.png", dpi=300)
    plt.close()

    # Plot 3: Absolute
    plt.figure(figsize=(8, 5))
    sns.barplot(data=top_corr, x="abs_corr", y="feature_name", palette="viridis")
    plt.title("Top 10 TSFresh Features Correlated with UPDRS Score (|r|)", weight="bold")
    plt.xlabel("Correlation Coefficient (|r|)")
    plt.ylabel("Feature Name")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "UPDRS_Correlation_Absolute.png", dpi=300)
    plt.close()
    
    print(" Created UPDRS correlation plots.")
except Exception as e:
    print(f"Error computing UPDRS correlations: {e}")

print(" Done generating visual summaries.")
