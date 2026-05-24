# 🏐 CT Guanabara Futevolei — Data Science Performance Analysis

> **University Extension Project V** | Descomplica — Data Science Program  
> **Author:** Filippo Alvim Cupolillo Bruno  
> **Partner Organization:** CT Guanabara — Futevolei Training Center  
> **Responsible:** Prof. Vitor Hugo Baptista Ribeiro  
> **Location:** Niterói — RJ, Brazil | **Period:** March–May 2026

---

## 📌 Overview

This project applies **Data Science and Machine Learning** to improve athletic performance at **CT Guanabara**, a Futevolei (foot-volleyball) training center located in Niterói, Rio de Janeiro.

Futevolei is a Brazilian beach sport that combines football technique with volleyball rules — played exclusively on sand, without the use of hands.

A full data pipeline was implemented — from standardized field collection to predictive modeling — generating **individualized training recommendations** for 5 pilot athletes over an 8-week period.

---

## 🎯 Key Results

| Metric | Pre (avg) | Post (avg) | Δ Improvement |
|---|---|---|---|
| Jump Height (cm) | 54.2 | 60.4 | **+11.4%** |
| Ball Control (touches) | 44.4 | 52.4 | **+18.0%** |
| Serve Accuracy (%) | 70.6% | 78.8% | **+11.6%** |
| Reception Efficiency (%) | 71.4% | 79.2% | **+10.9%** |
| Attack Rate (%) | 64.2% | 72.6% | **+13.1%** |
| **IPG (Overall Index)** | **0.661** | **0.745** | **+12.7%** |

**Model Performance:** Random Forest Regressor | R² = 0.81 | RMSE = 0.034

---

## 🗂️ Repository Structure

```
ct-guanabara-futevolei-datascience/
│
├── 📁 data/
│   ├── athletes_profile.csv         # Athlete demographics and IMC
│   ├── pre_intervention.csv         # Weeks 1–4 performance metrics
│   ├── post_intervention.csv        # Weeks 5–8 performance metrics
│   └── weekly_training_log.csv      # Session frequency, load & focus area
│
├── 📁 notebooks/
│   ├── 01_exploratory_analysis.ipynb    # EDA, correlations, distributions
│   ├── 02_data_cleaning.ipynb           # IQR outlier detection, imputation
│   ├── 03_modeling.ipynb                # Random Forest + cross-validation
│   └── 04_results_dashboard.ipynb       # Pre/post comparison + visualizations
│
├── 📁 src/
│   ├── data_pipeline.py             # Load, clean, IPG computation
│   ├── feature_engineering.py       # Lag features, load features, encoding
│   └── model.py                     # RF training, CV, feature importance plots
│
├── 📁 outputs/
│   ├── figures/                     # All generated charts (.png)
│   ├── clean_full_timeline.csv      # Cleaned merged dataset
│   ├── final_results_summary.csv    # Pre/post comparison per athlete
│   ├── pre_summary_stats.csv        # Descriptive statistics
│   └── cv_metrics.json              # Model cross-validation results
│
├── requirements.txt
└── README.md
```

---

## 🧪 Methodology — CRISP-DM Adapted

```
1. COLLECTION   →  Standardized field assessments (jump sensor, video, manual count)
2. INGESTION    →  CSV structuring with pandas DataFrames
3. CLEANING     →  IQR outlier detection, median imputation, dtype enforcement
4. EDA          →  Distributions, correlations, boxplots, heatmaps (Matplotlib/Seaborn)
5. MODELING     →  Random Forest Regressor, 5-fold cross-validation (Scikit-learn)
6. DELIVERY     →  Dashboard, individualized reports, coach recommendations
```

---

## 📐 General Performance Index (IPG)

The IPG is a composite metric aggregating all 5 performance dimensions:

```
IPG = 0.25 × JS_norm + 0.20 × BC_norm + 0.20 × SA + 0.20 × RE + 0.15 × AT

Where:
  JS  = Jump Height     (normalized: JS / 80cm)
  BC  = Ball Control    (normalized: BC / 80 touches)
  SA  = Serve Accuracy  (0–1 scale)
  RE  = Reception Eff.  (0–1 scale)
  AT  = Attack Rate     (0–1 scale)
```

---

## 🤖 Model Details

| Parameter | Value |
|---|---|
| Algorithm | Random Forest Regressor |
| n_estimators | 100 |
| max_depth | 5 |
| Target variable | Δ IPG (%) — week-over-week change |
| Evaluation | 5-Fold Cross-Validation |
| **R² (validation)** | **0.81** |
| **RMSE (validation)** | **0.034 (3.4%)** |

### Feature Importance

| Rank | Feature | Importance |
|---|---|---|
| 1 | Training frequency (sessions/week) | 38.2% |
| 2 | Previous week IPG | 30.7% |
| 3 | Rest days | 18.9% |
| 4 | BMI | 12.2% |

---

## 👥 Athletes

| Athlete | Position | Level | IPG Pre | IPG Post | Δ |
|---|---|---|---|---|---|
| Italo Ferreira | Attack | Intermediate | 0.660 | 0.750 | +13.6% |
| Elana Ferreira | Setting | Beginner | 0.603 | 0.680 | +12.8% |
| Ingrid Pontes | Defense | Advanced | 0.713 | 0.789 | +10.7% |
| Felipe Silva | Attack | Intermediate | 0.685 | 0.773 | +12.9% |
| Filippo A. C. Bruno | Versatile | Beginner | 0.643 | 0.732 | **+13.8%** |

---

## ⚙️ Setup & Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/ct-guanabara-futevolei-datascience.git
cd ct-guanabara-futevolei-datascience

# Create virtual environment
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt

# Run notebooks (in order)
jupyter notebook notebooks/
```

---

## 📦 Running the Source Scripts Directly

```bash
# From the src/ directory:
cd src

# Full pipeline
python data_pipeline.py

# Feature engineering
python feature_engineering.py

# Model training + plots
python model.py
```

---

## 📊 Generated Outputs

All charts are saved to `outputs/figures/`:

| File | Description |
|---|---|
| `baseline_metrics.png` | Week 1 baseline bar charts per athlete |
| `distributions.png` | Histogram + KDE for all metrics |
| `correlation_heatmap.png` | Pearson correlation matrix |
| `boxplots_pre_post.png` | Pre vs post boxplot comparison |
| `sessions_vs_ball_control.png` | Key correlation scatter plot |
| `feature_correlations.png` | Feature-target correlation bars |
| `cv_results.png` | Cross-validation RMSE and R² per fold |
| `feature_importance.png` | Random Forest feature importance |
| `actual_vs_predicted.png` | Scatter: actual vs predicted Δ IPG |
| `ipg_timeline.png` | 8-week IPG evolution for all athletes |
| `pre_vs_post_all_metrics.png` | Full metric comparison dashboard |
| `ipg_delta_ranking.png` | Athletes ranked by IPG improvement |
| `radar_charts.png` | Individual radar charts per athlete |

---

## 🏛️ Academic Context

- **Institution:** Centro Universitário União das Américas Descomplica
- **Program:** Bachelor's in Data Science
- **Project Type:** University Extension Project V (PEX V)
- **Partner CNPJ:** 40.298.588/0001-26

---

## 📄 License

This project is for academic and educational purposes.  
Data is synthetic and generated to match a real-world pilot study conducted at CT Guanabara, Niterói — RJ, Brazil.

---

*"We went from subjective perception to data-driven decisions. The athletes were motivated to see their evolution represented in charts and tables."*  
— Prof. Vitor Hugo Baptista Ribeiro, CT Guanabara Director
