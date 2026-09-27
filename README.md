# Reinforcement Learning Trees (RLT) — Comprehensive Analysis

A from-scratch implementation and benchmarking study of **Reinforcement Learning Trees**, following the **CRISP-DM** methodology (Cross-Industry Standard Process for Data Mining).

## 📋 Overview

This project implements RLT — an ensemble tree method that uses variable muting (importance-based feature selection at each split) and optional linear-combination splits — entirely from scratch, then evaluates it through a two-phase experimental design rather than a single benchmark pass.

## 🔬 Methodology

The study is split into two parts:

1. **Evaluation on Synthetic Scenarios** — systematic testing of RLT configuration combinations (ensemble type, embedded model, mutation rate, linear vs. axis-aligned splits) across four distinct synthetic scenarios, to identify the best-performing configuration for each type of data-generating process.
2. **Application on Real Datasets** — the winning configurations from phase 1 are mapped onto **10 real UCI datasets** (covering both classification and regression) and compared against established baselines.

This mirrors the CRISP-DM phases: Business Understanding → Data Understanding → Data Preparation → Modeling → Evaluation → Deployment (reporting).

## 🧠 Implementation

- **`Node`** — custom tree node supporting both univariate and linear-combination splits (`linear_coef`), with per-node tracking of variables used (for variable-importance-based muting).
- **`RLTTree`** — the core from-scratch tree learner. Key parameters:
  - `max_depth`, `min_samples_split`
  - `task` (`classification` / `regression`)
  - `max_vars` — number of variables sampled as split candidates at each node
  - `vi_threshold` — variable-importance threshold driving **variable muting** (down-weighting/excluding variables already found uninformative)
  - `linear_split` — enables linear-combination splits instead of single-variable splits
- **`RLTForest`** — bagged ensemble of `RLTTree` instances (0.7 subsample fraction per tree, `n_trees` configurable).

## 📊 Datasets

10 UCI datasets spanning both **classification** and **regression** tasks (e.g. Boston Housing, Parkinsons), chosen to stress-test RLT under varied sample sizes, feature/sample ratios, class imbalance, and correlation structure.

## ⚖️ Baselines Compared

- Random Forest
- Extra Trees
- BART (Bayesian Additive Regression Trees)
- Lasso

## 🛠️ Tech Stack

Python · NumPy · Pandas · scikit-learn · Matplotlib / Seaborn · (developed on Google Colab)

## 🚀 Running the Notebook

The notebook was developed and run on **Google Colab**, with datasets loaded from a personal Google Drive folder (`/content/drive/MyDrive/uci-datasets/`). To reproduce:

1. Download the 10 UCI datasets referenced in the notebook's data-loading cell.
2. Update `DATASETS_PATH` to point to wherever you store them locally (or re-mount your own Drive folder if running in Colab).
3. Run all cells top to bottom — the notebook installs its own extra dependencies (`plotly`, `xgboost`, `lightgbm`) in the first setup cell.

## 📈 Results

Full comparative results tables, per-dataset performance breakdowns, and visualizations (feature importance, scenario-configuration heatmaps, etc.) are generated inline in the notebook — see `RLT_analysis.ipynb`.

---

*Academic research project — implementation and evaluation of RLT as described in the reference literature on reinforcement learning trees.*
