"""
create_demographics_table_figure.py
Generates a clean table figure for presentation (gender, age, group info)
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
TABLE_DIR = Path("outputs/tables")
FIG_DIR = Path("outputs/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Read demographics summary file
df = pd.read_csv(TABLE_DIR / "demographics_summary.csv")

# Rename columns for slide readability
df = df.rename(
    columns={
        "Label": "Group",
        "N": "Subjects",
        "Age_Mean": "Mean Age",
        "Age_SD": "SD",
        "Male_N": "Male",
        "Female_N": "Female",
    }
)

# Combine mean and SD into one column (Mean Age ± SD)
df["Mean Age (±SD)"] = df.apply(
    lambda x: f"{x['Mean Age']:.1f} ± {x['SD']:.1f}", axis=1
)
df = df[["Group", "Subjects", "Mean Age (±SD)", "Male", "Female"]]

# ---- Create table figure ----
fig, ax = plt.subplots(figsize=(6, 1.5))
ax.axis("off")

table = ax.table(
    cellText=df.values, colLabels=df.columns, cellLoc="center", loc="center"
)

# Styling
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.6)

# Header color
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_text_props(weight="bold", color="white")
        cell.set_facecolor("#1565C0")  # deep blue header
    else:
        cell.set_facecolor("#E3F2FD")  # light blue rows

# Save figure
out_path = FIG_DIR / "demographics_table.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight", transparent=True)
plt.close()

print(f"✅ Table figure saved as: {out_path}")
