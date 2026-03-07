"""
06_generate_methodology_diagrams.py
--------------------------------------------------------
Generates high-quality diagrams for the presentation:
1. Cross-Validation Strategy (Stratified 5-Fold)
2. Machine Learning Pipeline
--------------------------------------------------------
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path

FIG_DIR = Path("outputs/figures/methodology")
FIG_DIR.mkdir(parents=True, exist_ok=True)

def plot_cv_strategy():
    """
    Visualizes a 5-Fold Stratified Cross-Validation strategy.
    """
    n_splits = 5
    n_samples = 100
    
    # Create simple mock data: indices
    indices = np.arange(n_samples)
    
    # Plot setup
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Per fold
    fold_height = 1.0
    gap = 0.2
    
    # Colors
    color_train = "#66c2a5"  # Greenish
    color_test = "#fc8d62"   # Orangish
    
    for i in range(n_splits):
        # Y position for this fold
        y_pos = (n_splits - 1 - i) * (fold_height + gap)
        
        # Draw the full bar as Training first
        ax.add_patch(Rectangle((0, y_pos), n_samples, fold_height, 
                               color=color_train, label="Training Data" if i==0 else None))
        
        # Overlay the Test segment
        # In 5-fold, test is 1/5th. 
        # Fold 0: 0-20, Fold 1: 20-40, ...
        start = i * (n_samples / n_splits)
        width = n_samples / n_splits
        
        ax.add_patch(Rectangle((start, y_pos), width, fold_height, 
                               color=color_test, label="Test Data" if i==0 else None))
        
        # Label each fold
        ax.text(-5, y_pos + fold_height/2, f"Fold {i+1}", 
                va='center', ha='right', fontsize=12, fontweight='bold')

    # Formatting
    ax.set_xlim(-15, n_samples + 5)
    ax.set_ylim(-gap, n_splits * (fold_height + gap))
    ax.set_xlabel("Data Samples Index", fontsize=12)
    ax.set_title("Stratified 5-Fold Cross-Validation Strategy", fontsize=14, pad=20)
    ax.axis('off')
    
    # Legend
    handles, labels = ax.get_legend_handles_labels()
    # Deduplicate legend
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper center', 
              bbox_to_anchor=(0.5, -0.05), ncol=2, fontsize=12)
    
    plt.tight_layout()
    out_path = FIG_DIR / "cv_strategy_diagram.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Generated CV Diagram: {out_path}")

def plot_pipeline_schematic():
    """
    Draws a flowchart-like block diagram for the processing pipeline using Matplotlib patches.
    Flow: Raw Data -> Preprocessing (Scaling) -> Feature Selection -> Model Training -> Evaluation
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    
    steps = [
        "Raw Data\n(Features)", 
        "Splitting\n(Stratified K-Fold)", 
        "Preprocessing\n(StandardScaler)", 
        "Model Training\n(GridSearch CV)", 
        "Evaluation\n(Metrics & Confusion Matrix)"
    ]
    
    x_positions = np.linspace(0.1, 0.9, len(steps))
    y_pos = 0.5
    box_width = 0.14
    box_height = 0.3
    
    # Draw boxes and arrows
    for i, step in enumerate(steps):
        # Draw Box
        box = Rectangle((x_positions[i] - box_width/2, y_pos - box_height/2), 
                        box_width, box_height, 
                        facecolor="#8da0cb", edgecolor="black", lw=2, zorder=2,
                        label="Process Step" if i==0 else None)
        ax.add_patch(box)
        
        # Add Text
        ax.text(x_positions[i], y_pos, step, 
                ha="center", va="center", fontsize=10, 
                fontweight="bold", color="white", zorder=3)
        
        # Draw Arrow to next box (except last one)
        if i < len(steps) - 1:
            start_x = x_positions[i] + box_width/2
            end_x = x_positions[i+1] - box_width/2
            ax.arrow(start_x, y_pos, end_x - start_x - 0.01, 0, 
                     head_width=0.03, head_length=0.02, fc='k', ec='k', lw=1.5, zorder=1)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title("Machine Learning Pipeline Workflow", fontsize=16)
    
    out_path = FIG_DIR / "pipeline_workflow.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Generated Pipeline Diagram: {out_path}")

if __name__ == "__main__":
    print("[START] Generating Methodology Diagrams...")
    plot_cv_strategy()
    plot_pipeline_schematic()
