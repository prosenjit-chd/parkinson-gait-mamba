import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# --- Load Data ---
file_path = Path("outputs/tables/feature_updrs_correlation.csv")
df = pd.read_csv(file_path)

# Ensure proper sorting
df = df.sort_values("abs_corr", ascending=False)

# Pick top features
top_corr = df.head(10)

sns.set(style="whitegrid", font_scale=1.1)
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["axes.titlesize"] = 13

# --- 1️⃣ Signed correlation plot ---
plt.figure(figsize=(8, 5))
sns.barplot(
    data=top_corr,
    x="correlation",
    y="feature_name",
    palette="coolwarm",
    hue="correlation",
    dodge=False,
    legend=False,
)
plt.axvline(0, color="gray", linestyle="--", lw=1)
plt.title("Top 10 TSFresh Features Correlated with UPDRS Score (Signed)", weight="bold")
plt.xlabel("Correlation Coefficient (r)")
plt.ylabel("Feature Name")
plt.tight_layout()
plt.savefig("UPDRS_Correlation_Signed.png", dpi=300)
plt.close()

# --- 2️⃣ Absolute correlation plot ---
plt.figure(figsize=(8, 5))
sns.barplot(data=top_corr, x="abs_corr", y="feature_name", palette="viridis")
plt.title("Top 10 TSFresh Features Correlated with UPDRS Score (|r|)", weight="bold")
plt.xlabel("Correlation Coefficient (|r|)")
plt.ylabel("Feature Name")
plt.tight_layout()
plt.savefig("UPDRS_Correlation_Absolute.png", dpi=300)
plt.close()

print("✅ Two plots generated successfully:")
print("   • UPDRS_Correlation_Signed.png  (for appendix/report)")
print("   • UPDRS_Correlation_Absolute.png (for presentation slide)")
