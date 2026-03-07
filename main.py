"""
main.py
--------------------------------------------------------
Mother script for the Parkinson’s Gait Analysis Baseline Pipeline.
Runs all necessary Deliverable 1 and Deliverable 2 scripts sequentially.
--------------------------------------------------------
"""
import os
import sys
from pathlib import Path

def run_script(module_name):
    print(f"\n{'='*70}")
    print(f"[RUN] Running: {module_name}")
    print(f"{'='*70}")
    
    # Run the script module
    import subprocess
    result = subprocess.run([sys.executable, "-m", module_name])
    if result.returncode != 0:
        print(f"[ERROR] Error encountered while running {module_name}")
        sys.exit(result.returncode)

if __name__ == "__main__":
    print("==================================================================")
    print(" Parkinson's Gait Analysis Baseline & Models Pipeline (D1 & D2)")
    print("==================================================================\n")
    
    # Ensure working directory is set to root
    os.chdir(Path(__file__).resolve().parent)

    # Deliverable 1: Data Analysis & Feature Extraction
    run_script("scripts.01_demographics")
    run_script("scripts.02_examples_vgrf")
    run_script("scripts.03_tsfresh_features") # Previously 03_tsfresh_pca_tsne.py
    run_script("scripts.04_visualize_summary")
    run_script("scripts.05_feature_combination")
    
    # Presentations Diagrams
    run_script("scripts.06_generate_methodology_diagrams")
    
    # Deliverable 2: Baseline Models Training & Optimization
    run_script("scripts.08_optimized_models")
    run_script("scripts.09_comparative_plots")
    
    print("\n==================================================================")
    print(" ALL PIPELINE STEPS COMPLETED SUCCESSFULLY!")
    print("Outputs available in: outputs/figures/ and outputs/tables/")
    print("==================================================================")
