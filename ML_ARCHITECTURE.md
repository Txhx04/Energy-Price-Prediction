# Spain Energy Forecasting: Master ML Architecture Guide

## 1. Executive Summary

This document serves as the technical whitepaper outlining the **Time-Series Forecasting Paradigm** adopted for the Spain Energy Forecasting project. The holistic analytical approach utilizes **ensemble stacking** across a robust spectrum of supervised regression algorithms designed to navigate both stochastic volatility and deterministic repeating temporal signals inherent in power grids.

The macro architecture encompasses a **multi-model validation approach**, evaluating six isolated learners: XGBoost (gradient boosting), GRU (recurrent neural networks), Random Forest (tree ensemble), Decision Tree, Logistic Regression, and Support Vector Machines. This rigorous validation phase ultimately dictates the **ensemble voting strategy**, combining the output predictions of the **4 high-performing tested models** (Ensemble Aggregate, XGBoost, Random Forest, and GRU) to enhance structural robustness and hedge against individual algorithm overconfidence.

The target objective function relies on a **dual-target prediction** pipeline mapping vectors synchronously for parallel tasks:
1. **Grid Consumption (MW):** Representing macroscopic state demand.
2. **Day-Ahead Market Price (€/MWh):** Representing clearing price density.

---

## 2. Key Component Deep-Dives

### Data Ingestion & Preparation
We combine grid metrics and real-time planetary parameters to synthesize a rich situational vector constraint map.
* **National Grid Ingestion:** ENTSO-E API integration tracks real-time consumption and supply-side generation paradigms (including renewable mixes and load forecasts) mapped via `entsoe-py`.
  * **Reference:** [entsoe_fetcher.py](backend/entsoe_fetcher.py)
* **Meteorological Synthesis:** The open-source Open-Meteo ERA5 reanalysis interface manages weather data collection, executing deterministic boundary alignment via temporal interpolation (extrapolating point-coordinate spatial boundaries across Spain's centroid).
  * **Reference:** [weather_fetcher.py](backend/weather_fetcher.py)

### Feature Engineering Engine
Raw sequences undergo massive deterministic transformations ensuring model stability against extreme outliers.
* **Temporal Sequence Mappings:** Calculation of autoregressive lag features representing historical memory structures (1h, 2h, 6h, 24h, 48h, 168h lags).
* **Statistical Boundaries:** Smoothing extreme outliers using rolling statistics tracking volatility (24h, 72h, 168h moving averages and standard deviation/volatility limits).
* **Cyclical Encoding:** Preventing extreme gradient jumps at boundary transitions (e.g., 23:00 to 00:00) using continuous coordinate sin/cos transformations of hourly, daily, and monthly cycles (preserves continuous spherical distance).
* **Macro Patterns:** Interaction features explicitly mapping Temperature × Consumption coefficients alongside Renewable Supply Percentage ratios. Peak hour and seasonal indicators directly modulate base demand elasticity.
  * **Reference:** [02_feature_engineering.ipynb](02_feature_engineering.ipynb)

### Model Architectures & Validation
Multiple base paradigms capture orthogonal dimensions of the problem space.
* **GRU (Gated Recurrent Unit):** A memory-efficient RNN mapping temporal momentum through a recurrent 2-gate mechanism (reset/update)—highly responsive for 24-hour sequence autoregression avoiding vanishing gradients.
* **XGBoost:** Aggressive gradient boosting framework excelling at mapping non-linear tabular feature interactions, stabilized through heavy programmatic L1/L2 regularization to prevent decision surface over-granularity.
* **Random Forest:** Ensemble of isolated decorrelated decision trees via feature/sample bagging. Acts as the stable anchor due to its absolute robustness to the extreme spikes evident in raw energy metrics.
* **Baseline Architectures:** Decision Tree, Logistic LR, and SVM provide continuous performance floor monitoring. 
* **Validation Protocols:** Incorporates continuous automated hyperparameter tuning (learning rates, max topological tree depth, batch sizing) strictly monitored via progressive fold iterations mapping time-series specific early-stopping logic.
  * **References:** [models.py](backend/models.py) and [03_model_training.ipynb](03_model_training.ipynb)

### Ensemble Strategy
Outputs from superior branches are collapsed into the definitive predictive vector matrix.
* **Voting Mechanism:** Employs a linear summation proxy weighted systematically using cross-validation confidence thresholding, mapping inputs from the 3 leading base components dynamically.
* **Weighted Averaging vs Hard Voting:** Operates exclusively using float-weighted averaging rather than distinct classification logic. This establishes safety fallbacks and stabilizes prediction curves precisely at sequence inflection points where models diverge.
  * **Reference:** [ensemble.py](ensemble.py)

### Backend API & Inference
Serves inference predictions efficiently managing asynchronous payload handling.
* **Endpoints:** The `/predict` API orchestrates model memory deployments for batch historical validations mapping explicit payload data contracts.
* **Inference Operations:** Performs comprehensive batch processing for 24-hour ahead horizons, balancing serialization footprints. Manages isolated CPU/GPU memory paths conditionally mapped depending on hardware capabilities mapping heavy tensor components vs light sklearn nodes.
  * **Reference:** [api.py](backend/api.py)

### Frontend Visualization
Communicates structural insights safely to decision makers.
* **Charting:** Synthesizes historical vs generated temporal vectors natively scaling D3 integrations to render granular uncertainty intervals.
* **UI Workflows:** Direct model comparative UI displaying base vs aggregate predictive density logic spanning 24-hour matrices. Identifies real-time optimal windows.
  * **Reference:** [frontend/components](frontend/)

---

## 3. Data Pipeline Architecture

```mermaid
graph TD
    A1[ENTSO-E API] --> B(Feature Engineering)
    A2[Weather API (ERA5)] --> B
    B --> C[Dataset Alignment & Merging]
    C --> D[Chronological Split]
    D --> E(Train 88%)
    D --> F(Validation 8.8%)
    D --> G(Test 2.9%)
    
    E --> H{Training Loop}
    H --> I[6 Isolated Models]
    F --> I
    I --> J{Model Validation}
    J --> K[XGBoost]
    J --> L[Random Forest]
    J --> M[GRU]
    J --> N[Discarded Baselines]
    
    K --> O((Ensemble Voting))
    L --> O
    M --> O
    
    O --> P[REST API Server /predict]
    P --> Q[Frontend Decision Dashboard]
```

---

## 4. Performance Benchmarks

All models are subjected to rigorous mathematical scrutiny. Validation operations track precision strictly across non-shuffle time-sequence test parameters mapping January 2026 out-of-distribution metrics.

* **Model Accuracy Metrics:** Primary benchmarking strictly enforces MAE (Mean Absolute Error), RMSE (Root Mean Squared Error for peak penalty), and MAPE (Mean Absolute Percentage Error to gauge relative deviations regardless of base demand volume). Reference `validation/metrics_summary.csv` and `TEST_REPORT.md` for historical test bounds capturing exactly 2025-2026 grid stress sequences.
* **Inference Latency:** The synchronized model evaluation process across 4 aggregated architectures runs predictably in real-time constraint environments, balancing single-sample payload overhead versus parallel batch generation sequences inside `/predict`.
* **Data Throughput:** Engineering scripts calculate processing magnitudes up to thousands of synthesized dimensional array rows per hour, constrained exclusively by ENTSO-E dynamic web limitations.

---

## 5. Training Strategy
Training respects the strict causality of time series (no future leaking). We enforce strict contiguous historical tracking mapping constraints backward from validation checkpoints. 

* **Dataset Splits:** 
  * **Train:** (2019 to 2024-12-31) representing *88%* memory density.
  * **Val:** (2024-12-31 to 2025-12-31) representing *8.8%* hyperparameter tracking bounds.
  * **Test:** (2025-12-31 to 2026-04-14) representing *2.9%* unseen isolated scoring matrices.
* **Optimization & Selection:** Deploys grid search methodologies (and localized Bayesian proxies) mapping standard loss optimization (MAE/MSE depending on threshold sensitivity) using generalized Adam mapping and SGD variants. Final architectural structures depend exclusively on contiguous cross-validation fold stability mapping structural tracking via early-stopping mechanisms preventing catastrophic network overfitting.

---

## 6. Validated vs. Tested Models

### Validated Base Tier (6 Models)
Extensive experiments successfully trained, scaled, and cross-validated the following base regressors:
- **XGBoost (XGB)**
- **Gated Recurrent Unit (GRU)**
- **Random Forest (RF)**
- **Decision Tree (DT)**
- **Logistic Regressions (LR)**
- **Support Vector Machine (SVM)**

### Final Test Suite Deployment (4 Models)
Following exhaustive empirical performance metric analysis scoring against severe non-linear distribution gaps appearing sequentially post-2022 (e.g. erratic pricing behavior during systemic European energy strain), only structurally capable pipelines were promoted into localized endpoints natively via API deployment and UX configuration.
- **Ensemble (The mathematical weighted optimal)**
- **XGBoost**
- **Random Forest** 
- **GRU**

*Rationale for exclusion:* Simple regressive hyperplanes (LR), non-iterative branches (DT), and margin classifications natively mapping scaling limitations iteratively (SVM), failed to consistently synthesize extreme systemic price floor oscillations or anomalous weather/demand interaction patterns smoothly. Thus, they are preserved explicitly as baselines tracking minimal efficiency floors mapping out-of-distribution environments in offline analytics.
