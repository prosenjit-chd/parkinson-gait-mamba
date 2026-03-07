"""
scripts/09_comparative_plots.py
--------------------------------------------------------
Generate 10+ Comparative Diagrams for Deliverable 2 Models
Input: outputs/tables/results_deliverable2_final.csv
Output: outputs/figures/comparison/*.png
--------------------------------------------------------
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Config
BASE_DIR = Path(__file__).resolve().parent.parent
TAB_FILE = BASE_DIR / "outputs" / "tables" / "results_deliverable2_final.csv"
FIG_DIR = BASE_DIR / "outputs" / "figures" / "comparison"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Set Style
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams['figure.figsize'] = (10, 6)

def load_results():
    if not TAB_FILE.exists():
        raise FileNotFoundError(f"Missing results file: {TAB_FILE}")
    return pd.read_csv(TAB_FILE)

def plot_bar_metric(df, metric, title, filename):
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x="Input", y=metric, hue="Model", palette="viridis")
    plt.title(title)
    plt.ylim(0.5, 1.0) # Zoom in to relevant range
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_heatmap(df, metric, title, filename):
    pivot = df.pivot(index="Model", columns="Input", values=metric)
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu", linewidths=.5)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_radar(df, input_type, filename):
    """Radar chart for all models on a specific input."""
    subset = df[df["Input"] == input_type]
    labels = ["Accuracy", "Sensitivity", "Specificity", "AUC"]
    
    # Setup radar
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    
    for i, row in subset.iterrows():
        values = row[labels].values.tolist()
        values += values[:1]
        ax.plot(angles, values, label=row["Model"], linewidth=2)
        ax.fill(angles, values, alpha=0.1)
        
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    plt.title(f"Model Comparison - {input_type} Input")
    plt.legend(bbox_to_anchor=(1.1, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_scatter_tradeoff(df, filename):
    """Sensitivity vs Specificity Scatter"""
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x="Specificity", y="Sensitivity", 
                    hue="Model", style="Input", s=200, palette="deep")
    plt.title("Sensitivity vs Specificity Trade-off")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.xlim(0.3, 1.0)
    plt.ylim(0.3, 1.0)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_line_trend(df, metric, title, filename):
    """Line plot to show performance trend across L -> R -> LR"""
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="Input", y=metric, hue="Model", marker="o", linewidth=2.5, palette="tab10")
    plt.title(title)
    plt.ylim(0.5, 1.0)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def main():
    print("[START] Generating Comparison Diagrams...")
    df = load_results()
    
    # Diagram 1: Accuracy Bar Chart
    plot_bar_metric(df, "Accuracy", "Accuracy Comparison by Foot Input", FIG_DIR / "01_bar_accuracy.png")
    
    # Diagram 2: AUC Bar Chart
    plot_bar_metric(df, "AUC", "AUC Comparison by Foot Input", FIG_DIR / "02_bar_auc.png")
    
    # Diagram 3: Sensitivity Bar Chart
    plot_bar_metric(df, "Sensitivity", "Sensitivity Comparison by Foot Input", FIG_DIR / "03_bar_sensitivity.png")
    
    # Diagram 4: Specificity Bar Chart
    plot_bar_metric(df, "Specificity", "Specificity Comparison by Foot Input", FIG_DIR / "04_bar_specificity.png")
    
    # Diagram 5: Heatmap AUC
    plot_heatmap(df, "AUC", "AUC Heatmap (Model vs Input)", FIG_DIR / "05_heatmap_auc.png")
    
    # Diagram 6: Heatmap Accuracy
    plot_heatmap(df, "Accuracy", "Accuracy Heatmap (Model vs Input)", FIG_DIR / "06_heatmap_accuracy.png")
    
    # Diagram 7: Radar Chart (Combined LR)
    plot_radar(df, "LR", FIG_DIR / "07_radar_LR.png")
    
    # Diagram 8: Radar Chart (Left L)
    plot_radar(df, "L", FIG_DIR / "08_radar_L.png")
    
    # Diagram 9: Sensitivity vs Specificity Trade-off
    plot_scatter_tradeoff(df, FIG_DIR / "09_scatter_sens_spec.png")
    
    # Diagram 10: Performance Trend (AUC)
    plot_line_trend(df, "AUC", "AUC Trend Across Inputs", FIG_DIR / "10_line_auc_trend.png")

    # Bonus: Accuracy Trend
    plot_line_trend(df, "Accuracy", "Accuracy Trend Across Inputs", FIG_DIR / "11_line_acc_trend.png")

    print(f"[OK] Generated 11 diagrams in {FIG_DIR}")

if __name__ == "__main__":
    main()
