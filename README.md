```markdown
# 🧠 Parkinson’s Gait Analysis with Time-Series Machine Learning

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange)
![Time Series](https://img.shields.io/badge/Time--Series-Feature%20Engineering-green)
![Status](https://img.shields.io/badge/Project-Completed-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

This repository contains a **complete time-series machine learning pipeline** developed for the **FAU Pattern Recognition Lab – Time Series Project (WS 2025/26)** under **Dr. Tomás Arias Vergara**.

The project demonstrates how **raw sensor signals can be transformed into statistical features and used for machine learning classification**.

Although the dataset originates from biomedical research, the techniques used here are **general-purpose time-series analysis methods** applicable to:

- wearable sensor data
- IoT monitoring
- industrial signals
- activity recognition
- health analytics

---

# 🎯 Project Objective

The goal of this project is to build a **complete time-series data pipeline** that:

1. Processes raw gait signals
2. Extracts statistical features
3. Analyzes feature structure
4. Applies dimensionality reduction
5. Trains machine learning classifiers
6. Compares model performance

The final system evaluates whether **time-series statistical features can distinguish between two classes of gait signals**.

---

# 📊 Dataset

The project uses the **PhysioNet GaitPDB dataset**.

🔗 https://physionet.org/content/gaitpdb/1.0.0/

The dataset contains **vertical ground reaction force (VGRF)** signals recorded during walking.

### Dataset Summary

| Property             | Value                          |
| -------------------- | ------------------------------ |
| Subjects             | 165                            |
| Control subjects     | 72                             |
| Parkinson’s subjects | 93                             |
| Sampling frequency   | 100 Hz                         |
| Recording duration   | ~2 minutes                     |
| Signal type          | Vertical Ground Reaction Force |

Each subject contains **signals from both the left and right foot**, enabling gait pattern analysis.

---

# ⚙️ Project Workflow

The pipeline consists of several stages.
```

Raw VGRF Signals
│
▼
Signal Preprocessing
│
▼
Time-Series Feature Extraction (TSFresh)
│
▼
Feature Analysis
(variance, correlation)
│
▼
Dimensionality Reduction
(PCA / t-SNE)
│
▼
Machine Learning Models
│
▼
Model Evaluation
(Accuracy, AUC, Sensitivity, Specificity)

```

---

# 📈 Time-Series Feature Engineering

Raw signals are converted into numerical feature vectors using **TSFresh**.

### Sliding Window Configuration

| Parameter | Value |
|----------|------|
Sampling rate | 100 Hz |
Window size | 30 ms |
Step size | 15 ms |

Statistical features extracted include:

- mean
- standard deviation
- root mean square
- kurtosis
- skewness
- signal energy
- interquartile range
- zero crossings

Each subject is represented by **56 extracted features**.

Three feature sets are generated:

| Feature Set | Description |
|-------------|-------------|
L | Left foot features |
R | Right foot features |
LR | Combined left and right features |

---

# 📊 Feature Analysis

Several analyses were performed to understand the feature space.

### Feature Variance Analysis
Identifies the most informative features.

```

outputs/figures/top_features_variance_LR.png

```

### Feature Correlation Heatmap
Reveals redundancy between features.

```

outputs/figures/feature_corr_heatmap_LR.png

```

### Feature Distribution Visualization

```

outputs/figures/feature_mean_distribution_LR.png

```

---

# 📉 Dimensionality Reduction

To visualize feature separability, dimensionality reduction techniques were applied.

### PCA Visualization

```

outputs/figures/pca_L.png
outputs/figures/pca_R.png
outputs/figures/pca_LR.png

```

These plots illustrate the **structure of the feature space and class clustering**.

---

# 🤖 Machine Learning Models

Five machine learning algorithms were evaluated.

| Model |
|------|
Random Forest |
Support Vector Machine (Linear) |
Support Vector Machine (RBF) |
Extra Trees |
CatBoost |

Training was performed using **5-fold cross-validation**.

---

# 📊 Model Evaluation Metrics

Models were evaluated using the following metrics.

| Metric | Description |
|------|-------------|
Accuracy | Overall prediction correctness |
Sensitivity | True positive rate |
Specificity | True negative rate |
AUC | Area under ROC curve |

---

# 🏆 Final Model Performance

| Input | Model | Accuracy | Sensitivity | Specificity | AUC |
|------|------|------|------|------|------|
L | CatBoost | **0.800** | 0.882 | 0.694 | 0.832 |
LR | ExtraTrees | 0.758 | 0.849 | 0.639 | **0.874** |
LR | RandomForest | 0.770 | 0.882 | 0.625 | 0.840 |

Key observations:

- **CatBoost achieved the highest accuracy**
- **ExtraTrees achieved the best AUC**
- Tree-based models outperformed SVM models
- Combining signals improves classification stability

---

# 📁 Project Structure

```

parkinson_time_series_project/

data/
raw/
metadata/

scripts/
01_demographics.py
02_examples_vgrf.py
03_tsfresh_features.py
04_visualize_summary.py
05_feature_combination.py
06_generate_methodology_diagrams.py
08_optimized_models.py
09_comparative_plots.py

src/
io_physionet.py
preprocess.py
features_tsfresh.py
analysis_variance.py
viz.py

outputs/
figures/
tables/

main.py
requirements.txt
README.md

````

---

# 🚀 Running the Pipeline

Execute the entire workflow with:

```bash
python main.py
````

The pipeline automatically performs:

1. Data analysis
2. Feature extraction
3. Feature visualization
4. Model training
5. Result comparison

Outputs will be saved in:

```
outputs/figures/
outputs/tables/
```

---

# ⚙️ Installation

Create virtual environment:

```bash
python -m venv .venv
```

Activate environment:

Windows

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Main libraries used:

- pandas
- numpy
- scikit-learn
- tsfresh
- matplotlib
- seaborn
- catboost

---

# 🌐 Data Source

PhysioNet Gait Database

Goldberger AL, Amaral LAN, Glass L, et al.

[https://physionet.org/content/gaitpdb/1.0.0/](https://physionet.org/content/gaitpdb/1.0.0/)

---

# 👨‍💻 Author

**Prosenjit Chowdhury**

M.Sc. Artificial Intelligence
Friedrich-Alexander University Erlangen-Nürnberg (FAU)

Working Student – Solution & Innovation Experience (SIX-SI)
SAP SE

GitHub
[https://github.com/prosenjit-chd](https://github.com/prosenjit-chd)

---

# 📜 Citation

If you use this repository in research, please cite:

```
Arias Vergara, T. (2025)
Time Series Intelligence Project – Parkinson’s Gait Analysis
Pattern Recognition Lab
Friedrich-Alexander University Erlangen-Nürnberg
```

```


```
