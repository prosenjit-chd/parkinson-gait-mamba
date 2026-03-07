"""
run_deliverable2.py
--------------------------------------------------------
Standalone script to run only the optimizations and 
comparisons for Deliverable 2 without re-running data 
processing steps.
--------------------------------------------------------
"""
import os
import sys
from pathlib import Path

def run_script(module_name):
    print(f"\n{'='*70}")
    print(f"[RUN] Running: {module_name}")
    print(f"{'='*70}")
    
    import subprocess
    result = subprocess.run([sys.executable, "-m", module_name])
    if result.returncode != 0:
        print(f"[ERROR] Error encountered while running {module_name}")
        sys.exit(result.returncode)

if __name__ == "__main__":
    print("==================================================================")
    print(" Deliverable 2: Baseline Models Training & Optimization")
    print("==================================================================\n")
    
    os.chdir(Path(__file__).resolve().parent)
    
    # Deliverable 2: Baseline Models Training & Optimization
    run_script("scripts.08_optimized_models")
    run_script("scripts.09_comparative_plots")
    
    print("\n==================================================================")
    print(" DELIVERABLE 2 STEPS COMPLETED SUCCESSFULLY!")
    print("Outputs available in: outputs/figures/ and outputs/tables/")
    print("==================================================================")
