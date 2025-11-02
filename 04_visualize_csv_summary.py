"""
04_visualize_csv_summary.py
Creates visual summaries from final CSVs for presentation.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

TABLE_DIR = Path("outputs/tables")
FIG_DIR = Path("outputs/figures")

FIG_DIR.mkdir(exist_ok=True, parents=True)

# ------------------------------------------------------------
# 1. DEMOGRAPHICS SUMMARY PLOT
# ------------------------------------------------------------
df_demo = pd.read_csv(TABLE_DIR / "demographics_summary.csv")

# Bar chart: number of patients vs controls
plt.figure(figsize=(6, 4))
sns.barplot(x="Label", y="N", data=df_demo, palette="Blues_d")
plt.title("Number of Subjects per Group")
plt.xlabel("Group")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(FIG_DIR / "subjects_per_group.png", dpi=150)
plt.close()

# ------------------------------------------------------------
# 2. FEATURE VARIANCE DISTRIBUTION (for LR combined)
# ------------------------------------------------------------
df_feat = pd.read_csv(TABLE_DIR / "features_LR.csv", index_col=0)

# Compute variance across features
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

# ------------------------------------------------------------
# 4. OPTIONAL: COMPARISON OF LEFT VS RIGHT FEATURE MEANS
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

print("✅ New summary plots saved in outputs/figures/")
