# Spain Energy Forecasting: Feature Inventory

This document tracks the features generated in the engineering pipeline and utilized by the machine learning models.

## 1. Feature Categories

| Category | Count | Purpose | Reference |
| :--- | :--- | :--- | :--- |
| **Original Features** | 32 | Raw data from ENTSO-E and Open-Meteo. | `fetch_spain_gap.py` |
| **Lag Features** | 18 | Autoregressive components (1h to 168h). | `02_feature_engineering.ipynb` |
| **Rolling Statistics** | 16 | Trend and volatility detection (24h to 168h). | `02_feature_engineering.ipynb` |
| **Derived Features** | 12 | Interactions, rate of change, and domain logic. | `02_feature_engineering.ipynb` |
| **Cyclical Encoding** | 8 | Continuous temporal distance mapping. | `02_feature_engineering.ipynb` |
| **Target Variables** | 48 | H+1 to H+24 horizons for Consumption and Price. | `02_feature_engineering.ipynb` |

---

## 2. Key High-Impact Features

### Autoregressive Lags
*   `consumption_lag_24h`, `consumption_lag_168h`: Daily and weekly seasonality indicators.
*   `price_lag_24h`, `price_lag_168h`: Market rhythm anchors.

### Rolling Windows (Stability Features)
*   `consumption_rolling_mean_24h`: Smooths hourly fluctuations for 24h-ahead trend.
*   `price_rolling_std_24h`: Captures price volatility for risk-adjusted scheduling.

### Domain Interactions
*   `temp_consumption_interaction`: Models the non-linear relationship between temperature and load.
*   `renewable_percentage`: Impacts the merit order curve and final market pricing.
*   `is_peak_hour`: Binary indicator for high-demand business hours (9am-9pm).

### Cyclical Transformations
*   `hour_sin`, `hour_cos`: Ensures 23:00 and 00:00 remain mathematically adjacent.
*   `dayofweek_sin`, `dayofweek_cos`: Captures the weekend/weekday transition without ordinal bias.

---

## 3. Model Input Mapping

| Model | Input Feature Vector | Input Shape |
| :--- | :--- | :--- |
| **XGBoost / RF** | Full Tabular Vector (excluding targets) | `[batch_size, 82]` |
| **GRU / LSTM** | Scaled Sequences (sliding window) | `[batch_size, 48, 82]` |

*Note: The input dimensionality of 82 features excludes the 48 target columns generated in the end of the pipeline.*

---

## 4. Pipeline Reference
| Stage | Source Notebook / Script | Output |
| :--- | :--- | :--- |
| **Ingestion** | `fetch_spain_gap.py` | `outputs/spain_gap_2019_2026.csv` |
| **Engineering** | `02_feature_engineering.ipynb` | `Data/features/spain_features_test.csv` |
| **Training** | `03_model_training.ipynb` | `outputs/models/*.pkl`, `*.pt` |
| **Inference** | `backend/models.py` | Live Predictions |
