# 🧠 Parkinson’s Disease Detection from Gait Signals using Time-Series Machine Learning

This repository contains a **complete time-series machine learning pipeline** developed for the **FAU Pattern Recognition Lab – Time Series Intelligence Project (WS 2025/26)** under **Dr. Tomás Arias Vergara**.

The project demonstrates how **raw gait sensor signals can be transformed into statistical feature representations and used for machine learning classification**.

Although the dataset originates from biomedical research, the techniques implemented here represent **general-purpose time-series analysis methods** applicable to:

- wearable sensor data
- IoT monitoring systems
- industrial sensor signals
- human activity recognition
- digital health analytics

---

# 🎯 Project Objective

The goal of this project is to build a **complete end-to-end time-series machine learning pipeline** that:

1. Processes raw gait signals
2. Extracts statistical features from time-series data
3. Analyzes feature structure and redundancy
4. Applies dimensionality reduction techniques
5. Trains machine learning classifiers
6. Compares model performance across feature configurations

The final system evaluates whether **time-series statistical representations of gait signals can distinguish Parkinson’s patients from healthy controls**.

---

# 📊 Dataset

The project uses the **PhysioNet GaitPDB dataset**.

🔗 https://physionet.org/content/gaitpdb/

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

Each subject contains **signals from both left and right feet**, enabling gait asymmetry analysis.

---

# ⚙️ Project Workflow

The pipeline follows a typical **time-series machine learning workflow**.

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

| Parameter     | Value  |
| ------------- | ------ |
| Sampling rate | 100 Hz |
| Window size   | 30 ms  |
| Step size     | 15 ms  |

Statistical features extracted include:

- mean
- standard deviation
- root mean square
- kurtosis
- skewness
- signal energy
- interquartile range
- zero crossings

Each subject is represented by **56 extracted statistical features**.

Three feature sets are generated:

| Feature Set | Description                      |
| ----------- | -------------------------------- |
| **L**       | Left foot features               |
| **R**       | Right foot features              |
| **LR**      | Combined left and right features |

---

# 📊 Feature Analysis

Several analyses were performed to understand the **structure and quality of extracted features**.

### Feature Variance Analysis

Identifies the most informative features.

```

outputs/figures/top_features_variance_LR.png

```

### Feature Correlation Heatmap

Reveals redundancy between extracted features.

```

outputs/figures/feature_corr_heatmap_LR.png

```

### Feature Distribution Visualization

```

outputs/figures/feature_mean_distribution_LR.png

```

---

# 📉 Dimensionality Reduction

Dimensionality reduction techniques were applied to visualize the **feature space structure and class separability**.

### PCA Visualization

```

outputs/figures/pca_L.png
outputs/figures/pca_R.png
outputs/figures/pca_LR.png

```

These plots illustrate clustering patterns between **Parkinson’s disease (PD) and control subjects**.

---

# 🤖 Machine Learning Models

Five machine learning algorithms were evaluated.

| Model                                  |
| -------------------------------------- |
| Random Forest                          |
| Support Vector Machine (Linear Kernel) |
| Support Vector Machine (RBF Kernel)    |
| Extra Trees                            |
| CatBoost                               |

Training was performed using **stratified 5-fold cross-validation**.

---

# 📊 Model Evaluation Metrics

Models were evaluated using the following metrics.

| Metric      | Description                       |
| ----------- | --------------------------------- |
| Accuracy    | Overall prediction correctness    |
| Sensitivity | True positive rate (PD detection) |
| Specificity | True negative rate                |
| AUC         | Area under ROC curve              |

---

# 🏆 Baseline Model Performance

The table below summarizes baseline performance across all models and feature configurations.

<p align="center">
<img src="outputs/figures/baseline_performance_summary.png" width="900">
</p>

### Key Observations

- **CatBoost achieves the highest single-input accuracy (0.80)** using left-foot features
- **ExtraTrees with combined inputs achieves the highest AUC (0.874)**
- **Tree-based ensemble models outperform SVM models** across most metrics
- **Combining left and right signals improves classification robustness**

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

```

---

# 🚀 Running the Pipeline

Execute the entire workflow with:

```bash
python main.py
```

The pipeline automatically performs:

1. Dataset analysis
2. Feature extraction
3. Feature visualization
4. Model training
5. Model comparison

Outputs are saved to:

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

Activate environment (Windows):

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

[https://physionet.org/content/gaitpdb/](https://physionet.org/content/gaitpdb/)

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

---
