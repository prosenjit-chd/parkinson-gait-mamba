# 🧠 Parkinson’s Gait Analysis – TSFresh Baseline & Mamba Prep (Python)

This repository contains the **Nov 7 Data Analysis milestone** for the FAU PRL time-series project:

- Load **PhysioNet Gaitpdb** vertical ground reaction force (VGRF) signals
- Summarize **demographics** (PD vs Control, age, sex)
- Plot **example VGRF** signals (Control vs PD)
- Extract **TSFresh** features (Right / Left / Combined)
- Explore separability via **PCA** and **t-SNE** (for Dec/Feb models)

The repo is organized so you can later add **baseline models (Dec 7)** and **Mamba/Selective-SSM models (Feb 20)** cleanly.

---

## 📁 Project Structure

```
parkinsons_mamba_project/
├── data/
│   ├── raw/                     # 312 PhysioNet .txt files (Ga*/Ju*/Si*)
│   └── metadata/
│       ├── demographics.xlsx    # OR demographics.txt / .html (auto-detected)
│       └── (other meta files)
├── outputs/
│   ├── figures/                 # auto-saved plots (png)
│   └── tables/                  # auto-saved CSVs
├── scripts/
│   ├── 01_demographics.py       # summary table + age histogram/boxplot
│   ├── 02_examples_vgrf.py      # example Control & PD VGRF plots
│   └── 03_tsfresh_pca_tsne.py   # TSFresh features + PCA/t-SNE (R/L/LR)
├── src/
│   ├── __init__.py
│   ├── config.py                # paths (data, outputs)
│   ├── io_physionet.py          # file listing + VGRF reader
│   ├── preprocess.py            # length standardization & normalization
│   ├── features_tsfresh.py      # TSFresh extraction utils
│   └── viz.py                   # plotting helpers (age + VGRF)
├── requirements.txt
└── README.md
```

> Put **all 312 `.txt`** signal files into `data/raw/`.
> Put **`demographics.xlsx`** (or `.txt`/`.html`) into `data/metadata/`.

---

## 🔧 Setup Instructions

### 🧬 Prerequisites

- Python ≥ **3.10**
- (Windows/Mac/Linux supported)

### ✅ Create environment & install deps

```bash
# from repo root
python -m venv .venv

# Windows
.\.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

# If demographics is .xlsx/.xls
pip install openpyxl xlrd
```

`requirements.txt` (already included):

```
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
scikit-learn>=1.3
tsfresh>=0.20
tqdm>=4.66
```

---

## 🚀 How to Run (Nov 7 Deliverables)

> Run **from the project root**. Use the module form to ensure imports work.

### 1) Demographics summary + age plots

```bash
python -m scripts.01_demographics
```

**Outputs**

- `outputs/tables/demographics_summary.csv`
- `outputs/tables/demographics_clean.csv`
- `outputs/figures/age_distribution.hist.png`
- `outputs/figures/age_distribution.box.png`

### 2) Example VGRF plots (Control vs PD)

```bash
python -m scripts.02_examples_vgrf
```

**Outputs**

- `outputs/figures/example_control.png`
- `outputs/figures/example_patient.png`

### 3) TSFresh features + PCA/t-SNE (Right / Left / Combined)

```bash
python -m scripts.03_tsfresh_pca_tsne
```

**Outputs**

- `outputs/tables/tsfresh_features_R.csv`
- `outputs/tables/tsfresh_features_L.csv`
- `outputs/tables/tsfresh_features_LR.csv`
- `outputs/figures/pca_R.png`, `tsne_R.png`
- `outputs/figures/pca_L.png`, `tsne_L.png`
- `outputs/figures/pca_LR.png`, `tsne_LR.png`

> **Note:** If you see a Windows `loky`/CPU core warning from `joblib`, it’s harmless. Features and plots are still generated.

---

## 🧩 Methods (brief)

- **Signals:** Vertical Ground Reaction Force (VGRF) — per foot totals (L_total, R_total)
- **Standardization:** Each sequence is trimmed/padded to fixed duration (e.g., 100 s @ 100 Hz) and **min-max normalized (per sequence)**
- **Features:** **TSFresh** EfficientFCParameters (hundreds of time-series stats per foot)
- **Visualization:** **PCA** and **t-SNE** (2D) to inspect PD vs Control separability
- **Best view (usually):** Combined **Right+Left** features show clearer separation than either foot alone.

---

## 🗺️ Roadmap to Next Milestones

### 📅 Dec 7 — Baseline Models

- Train on TSFresh features with **5-fold stratified CV**:

  - Random Forest
  - SVM (linear)
  - SVM (RBF)

- Report: **Accuracy, Sensitivity, Specificity, AUC**
- Provide **confusion matrices** (Left, Right, Combined)
- (If working in pairs) add **Linear-SVR** to predict **weight**:

  - Metrics: **MAE, MSE, Pearson r**
  - **Predicted vs Target** scatter plots (Left & Right best model)

### 📅 Feb 20 — Mamba / Selective-SSM Models

- Implement **Mamba (Selective SSM)** sequence models on the raw VGRF (or learned representations)
- Compare against baselines on identical folds/metrics
- Submit **final slides** (and **4-page report + 1 page references** for 10 ECTS)

---

## 🧪 Troubleshooting

- **`ModuleNotFoundError: No module named 'src'`**

  - Make sure `src/__init__.py` exists and run with `python -m scripts.XX` from the project root.

- **Excel read error (`xlrd`/`openpyxl`)**

  - `pip install openpyxl xlrd` and ensure the demographics file is in `data/metadata/`.

- **PCA error “Input X contains NaN”**

  - Already handled in `03_tsfresh_pca_tsne.py` (median imputation). Pull latest code.

- **No files found**

  - Ensure your `.txt` signals are directly in `data/raw/` (not nested).

---

## 🌐 Data Source

- PhysioNet Gaitpdb v1.0.0 — “Gait in Parkinson’s disease” (vertical ground reaction force)
  (Place all provided `.txt` files into `data/raw/`)

---

## 🤝 Contributing / Branching

- **main**: stable code for milestones
- **feat/**: new features (e.g., `feat/baselines`, `feat/mamba`)
- **fix/**: hotfixes
- Use conventional commits: `feat:`, `fix:`, `docs:`, `refactor:`, etc.

---

## 🗂️ .gitignore (add this to keep repo clean)

Create a `.gitignore` in the repo root:

```
# env & OS
.venv/
__pycache__/
*.pyc
.DS_Store

# IDE
.vscode/
.idea/

# data & outputs (keep structure, ignore big/raw)
data/raw/*
!data/raw/.gitkeep
outputs/*
!outputs/.gitkeep

# notebooks (if you add later)
*.ipynb_checkpoints/
```

_(Add empty `.gitkeep` files to keep folders in Git: `data/raw/.gitkeep`, `outputs/.gitkeep`)_

---

## 🧾 How to publish to GitHub

```bash
# from repo root
git init
git add .
git commit -m "init: Nov 7 data analysis milestone (demographics, VGRF plots, TSFresh + PCA/t-SNE)"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/parkinsons-mamba-gait.git
git push -u origin main
```

---

## 👨‍💻 Author

**Prosenjit Chowdhury**
M.Sc. Artificial Intelligence – FAU Erlangen-Nürnberg
Working Student, SIX SI - Proserv & EC&O Department, @ SAP-SE
🔗 GitHub: `@prosenjit-chd`

---

### If anything’s unclear about the _future tasks_:

- I’ve already mapped your **Dec 7** and **Feb 20** deliverables into a clean roadmap.
- When you’re ready, I’ll add **scripts/04_baselines.py** (RF, SVMs, metrics + confusion matrices) and a **Mamba training module** stub so you can iterate fast.
