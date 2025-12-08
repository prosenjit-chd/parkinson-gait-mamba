import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import os
import matplotlib.pyplot as plt
import seaborn as sns

# === File Paths ===
features_path = "outputs/tables/features_LR.csv"
updrs_path = "data/metadata/demographics.xls"
output_corr_csv = "outputs/tables/feature_updrs_correlation.csv"
output_corr_fig = "outputs/figures/updrsFeatureCorrelationVisualization.png"

print("📂 Loading features and UPDRS data...")

# === Load features ===
features_df = pd.read_csv(features_path)
print(f"✅ Features loaded: {features_df.shape}")

# === Load UPDRS file ===
updrs_df = pd.read_excel(updrs_path)
print(f"✅ UPDRS metadata loaded: {updrs_df.shape}")

# === Clean IDs ===
features_df["subject_clean"] = features_df["subject"].str.extract(r"([A-Za-z]+\d+)")
updrs_df["ID_clean"] = updrs_df["ID"].astype(str).str.strip()

# === Convert UPDRS column to numeric ===
updrs_df["UPDRS"] = pd.to_numeric(updrs_df["UPDRS"], errors="coerce")

# === Merge features with UPDRS ===
merged = pd.merge(
    features_df, updrs_df, left_on="subject_clean", right_on="ID_clean", how="inner"
)

# Drop NaNs
merged = merged.dropna(subset=["UPDRS"])
print(f"✅ Clean merged shape: {merged.shape}")
print(f"UPDRS range: {merged['UPDRS'].min()} – {merged['UPDRS'].max()}")

# === Compute correlations ===
corr_list = []
for feature in features_df.columns:
    if feature in ["subject", "subject_clean"]:
        continue
    if merged[feature].nunique() > 1:  # ignore constant features
        corr, _ = pearsonr(merged[feature], merged["UPDRS"])
        corr_list.append((feature, corr))

# === Create DataFrame ===
df_corr = pd.DataFrame(corr_list, columns=["feature_name", "correlation"])
df_corr["abs_corr"] = df_corr["correlation"].abs()
df_corr = df_corr.sort_values(by="abs_corr", ascending=False)

# === Save correlation results ===
os.makedirs(os.path.dirname(output_corr_csv), exist_ok=True)
df_corr.to_csv(output_corr_csv, index=False)
print(f"✅ Saved feature–UPDRS correlation results to: {output_corr_csv}")

# === Plot Top 10 Correlated Features ===
top_corr = df_corr.head(10)
plt.figure(figsize=(8, 5))
sns.barplot(x="correlation", y="feature_name", data=top_corr, palette="viridis")
plt.title(
    "Top 10 TSFresh Features Correlated with UPDRS Score", fontsize=13, weight="bold"
)
plt.xlabel("Correlation Coefficient (r)")
plt.ylabel("Feature Name")
plt.tight_layout()

# Save figure
os.makedirs(os.path.dirname(output_corr_fig), exist_ok=True)
plt.savefig(output_corr_fig, dpi=300)
plt.show()

print(f"📊 Figure saved to: {output_corr_fig}")
print("✅ Done — Real correlations computed successfully!")
