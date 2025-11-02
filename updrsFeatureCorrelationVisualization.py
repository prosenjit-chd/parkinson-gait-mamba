import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Simulated example: 10 features with random correlations
np.random.seed(42)
features = [f"feature_{i}" for i in range(1, 11)]
corr_values = np.sort(np.random.uniform(0.2, 0.9, 10))[::-1]

df = pd.DataFrame({"Feature": features, "Correlation": corr_values})

# Bar plot
plt.figure(figsize=(8, 4))
sns.barplot(x="Correlation", y="Feature", data=df, palette="viridis", edgecolor="black")
plt.title(
    "Top 10 Features Correlated with UPDRS Score", fontsize=13, weight="bold", pad=12
)
plt.xlabel("Correlation Coefficient (|r|)")
plt.ylabel("Feature Name")
plt.grid(axis="x", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()
