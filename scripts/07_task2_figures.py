import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ---------------------------
# Load baseline results from JSON
# ---------------------------
df = pd.read_json("../outputs/tables/baseline_results.json")

# Each cell is a dict → we expand it
clean_rows = []

for model_name in df.index:
    for foot in df.columns:
        metrics = df.loc[model_name, foot]
        clean_rows.append(
            {
                "Foot": foot,
                "Model": model_name,
                "Accuracy": metrics["Accuracy"],
                "Sensitivity": metrics["Sensitivity"],
                "Specificity": metrics["Specificity"],
                "AUC": metrics["AUC"],
            }
        )

clean_df = pd.DataFrame(clean_rows)

# Ensure ordering
order = [
    ("LEFT", "SVM_linear"),
    ("LEFT", "SVM_rbf"),
    ("LEFT", "RandomForest"),
    ("RIGHT", "SVM_linear"),
    ("RIGHT", "SVM_rbf"),
    ("RIGHT", "RandomForest"),
    ("COMBINED", "SVM_linear"),
    ("COMBINED", "SVM_rbf"),
    ("COMBINED", "RandomForest"),
]

clean_df["order"] = clean_df.apply(
    lambda r: order.index((r["Foot"], r["Model"])), axis=1
)
clean_df = clean_df.sort_values("order")

# ---------------------------
# Output folder
# ---------------------------
FIG_DIR = Path("../outputs/figures/task2_clean/")
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------
# Plot function (9 clean bars)
# ---------------------------
def plot_metric(metric_name, ylabel):
    plt.figure(figsize=(10, 6))

    x_labels = clean_df.apply(lambda r: f"{r['Foot']}\n{r['Model']}", axis=1)
    values = clean_df[metric_name].values

    plt.bar(np.arange(len(values)), values)
    plt.xticks(np.arange(len(values)), x_labels, rotation=45, ha="right")
    plt.ylabel(ylabel)
    plt.title(f"{metric_name} Comparison (Clean 9-Bar Plot)")
    plt.tight_layout()

    out_path = FIG_DIR / f"{metric_name.lower()}_clean.png"
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"✅ Saved {metric_name}: {out_path}")


# ---------------------------
# Generate all 4 metric plots
# ---------------------------
plot_metric("Accuracy", "Accuracy")
plot_metric("Sensitivity", "Sensitivity")
plot_metric("Specificity", "Specificity")
plot_metric("AUC", "AUC")

print("\n🎉 All 9-bar clean figures generated successfully!")
