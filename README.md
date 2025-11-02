# 🧠 Parkinson’s Gait Analysis – TSFresh Baseline & Mamba Prep (Python)

This repository contains the **Nov 7 Data Analysis milestone** for the FAU Pattern Recognition Lab (Time Series Intelligence WS 2025/26) project under **Dr. Tomás Arias Vergara**.

The goal is to build a complete data-to-model pipeline for **automatic Parkinson’s Disease (PD) detection** from **Vertical Ground Reaction Force (VGRF)** signals, beginning with traditional feature engineering and progressing toward advanced **State Space Models (SSMs)** such as **Mamba**.

---

## 🎯 Project Overview

**Tasks completed in the Nov 7 milestone:**

1. Load and clean the **PhysioNet Gaitpdb dataset** (PD vs Control).
2. Summarize **demographics** — subject count, age, and gender.
3. Visualize **example VGRF signals** for Control vs PD.
4. Extract **statistical features with TSFresh** (Left, Right, Combined).
5. Analyze **feature variance & correlation structure**.
6. Combine **Left & Right foot features** via concatenation/averaging.
7. Compute **feature–UPDRS correlations** to interpret disease severity links.
8. Apply **PCA** for dimensionality reduction and visualize PD vs Control separability.
9. Summarize all outputs (CSV + figures) for future baseline model training.

Upcoming deliverables include baseline ML models (Dec 7) and Mamba SSMs (Feb 20).

---

## 📁 Project Structure

```
parkinsons_mamba_project/
├── data/
│   ├── raw/                     # 312 PhysioNet .txt files (Ga*/Ju*/Si*)
│   └── metadata/
│       ├── demographics.xlsx    # subject age/sex/group info
│       └── (other meta files)
├── outputs/
│   ├── figures/                 # all generated plots (.png)
│   └── tables/                  # all CSV outputs
├── scripts/
│   ├── 01_demographics.py       # summary table + boxplot/histogram
│   ├── 02_examples_vgrf.py      # example Control & PD VGRF signals
│   ├── 03_tsfresh_features.py   # TSFresh feature extraction (L/R)
│   ├── 04_visualize_summary.py  # variance, correlation, PCA, UPDRS plots
│   └── 05_feature_combination.py# left-right concatenation diagram
├── src/
│   ├── __init__.py
│   ├── config.py                # all folder paths
│   ├── io_physionet.py          # data reader / loader
│   ├── preprocess.py            # normalization & trimming
│   ├── features_tsfresh.py      # TSFresh utility functions
│   ├── analysis_variance.py     # variance/correlation helpers
│   └── viz.py                   # all plotting utilities
├── presentation/
│   ├── slides_nov7.pptx         # FAU PRL presentation
│   └── slides_nov7.pdf          # exported PDF version
├── requirements.txt
└── README.md
```

---

## 🧩 Methods Summary

### **Signals**

Vertical Ground Reaction Force (VGRF) from both feet
→ sampled at **100 Hz (10 ms intervals)**
→ ~120 s per subject

### **Demographics**

| Group   | Subjects | Mean Age (±SD) | Male | Female |
| :------ | :------: | :------------: | :--: | :----: |
| Control |    72    |   63.7 ± 8.7   |  40  |   32   |
| PD      |    93    |   66.3 ± 9.5   |  58  |   35   |

Balanced gender and similar age ranges ensure fair comparability.

### **Feature Extraction (TSFresh)**

- Window = 30 ms , Step = 15 ms , Sampling = 100 Hz
- Extract 14 statistical features (mean, std, IQR, RMS, energy, kurtosis, etc.) per window
- Aggregate → 56 features per foot → 112 for combined (Left + Right)

### **Feature Combination Strategy**

Two approaches evaluated:

1. **Concatenation** → merge L + R feature vectors (112 features).
2. **Averaging** → mean of corresponding L/R features (56 features).
   Combined features capture inter-foot asymmetry — a key PD indicator.

### **Feature Variance & Correlation**

- Identified top 10 features by variance (`kurtosis_kurt`, `zero_crossings_kurt`, etc.).
- Correlation heatmap reveals low redundancy → diverse informative features.

### **Feature–UPDRS Correlation**

Computed Pearson correlation between each feature and UPDRS score.
Top correlated features highlight motor impairment patterns.

### **Dimensionality Reduction**

Applied PCA to reduce features → 2 principal components.
Partial PD vs Control separation visible, strongest in right-foot features.

---

## 🖼️ Key Figures Generated

| Category            | File Name                                           | Description                      |
| :------------------ | :-------------------------------------------------- | :------------------------------- |
| Demographics        | `age_boxplot.png`, `age_histogram.png`              | Age distribution plots           |
| Signal Examples     | `example_control.png`, `example_patient.png`        | VGRF pattern comparison          |
| Feature Extraction  | `tsfresh_features_LR.csv`                           | Combined feature table           |
| Variance Analysis   | `top_features_variance_LR.png`                      | Top 10 feature variances         |
| Correlation Heatmap | `feature_corr_heatmap_LR.png`                       | Feature relationships            |
| Feature Combination | `feature_combination_diagram.png`                   | L/R concatenation workflow       |
| UPDRS Analysis      | `updrs_feature_correlation.png`                     | Top 10 UPDRS-correlated features |
| PCA Visualization   | `pca_left.png`, `pca_right.png`, `pca_combined.png` | PD vs Control separation plots   |
| Summary             | `baseline_summary_diagram.png`                      | Pipeline overview diagram        |

---

## 🚀 How to Run (Nov 7 Deliverables)

Run modules from project root for consistent imports.

```bash
python -m scripts.01_demographics
python -m scripts.02_examples_vgrf
python -m scripts.03_tsfresh_features
python -m scripts.04_visualize_summary
python -m scripts.05_feature_combination
```

Outputs are automatically saved in `outputs/tables/` and `outputs/figures/`.

---

## 🗺️ Roadmap to Next Milestones

### 📅 **Dec 7 – Baseline Models**

Train on TSFresh features using 5-fold stratified CV:

- Random Forest
- SVM (linear & RBF)

**Metrics:** Accuracy, Sensitivity, Specificity, AUC
**Visuals:** Confusion matrices for Left, Right, Combined sets

### 📅 **Feb 20 – Mamba / Selective-SSM Models**

- Implement Mamba and Selective-SSM on raw VGRF signals.
- Compare against TSFresh baselines using identical folds and metrics.
- Deliver final slides + 4-page report (+ 1 reference page) for 10 ECTS submission.

---

## ⚙️ Setup Instructions

```bash
python -m venv .venv
.\.venv\Scripts\activate      # Windows
source .venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
```

**Requirements:**

```
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
seaborn>=0.12
scikit-learn>=1.3
tsfresh>=0.20
tqdm>=4.66
openpyxl
```

---

## 🧪 Troubleshooting

| Issue                                        | Fix                                                   |
| :------------------------------------------- | :---------------------------------------------------- |
| `ModuleNotFoundError: No module named 'src'` | Run scripts using `python -m scripts.xxx` from root   |
| Excel read error                             | Install `openpyxl` or `xlrd`                          |
| PCA error (contains NaN)                     | Handled by median imputation                          |
| No files found                               | Ensure `.txt` signals are in `data/raw/` (not nested) |

---

## 🌐 Data Source

[PhysioNet Gaitpdb v1.0.0](https://physionet.org/content/gaitpdb/1.0.0/) — Vertical Ground Reaction Force signals for Parkinson’s Disease and Healthy Controls.

---

## 👨‍💻 Author

**Prosenjit Chowdhury**
M.Sc. Artificial Intelligence — FAU Erlangen-Nürnberg
Working Student, SIX SI – Professional Services & EC&O, SAP SE
🔗 GitHub: [@prosenjit-chd](https://github.com/prosenjit-chd)

---

## 📊 Citation

If you use this repository, please cite:

> Arias Vergara T., Pattern Recognition Lab (2025): Time Series Intelligence – Parkinson’s Gait Analysis using SSMs. FAU Erlangen-Nürnberg.

---
