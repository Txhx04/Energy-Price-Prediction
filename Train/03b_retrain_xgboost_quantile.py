"""
PHASE 2a: RETRAIN XGBoost WITH QUANTILE LOSS
=============================================

Retrains XGBoost price prediction with quantile loss (tau=0.90)
Saves to: outputs/models_quantile/xgb_price_quantile.pkl

Goal: Improve HIGH-value prediction accuracy from 0.0% baseline
Does NOT modify existing outputs/models/xgb_price.pkl
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("PHASE 2a: XGBoost Quantile Loss Retraining")
print("=" * 80)

# ============================================================================
# 1. LOAD AND PREPROCESS TRAINING DATA
# ============================================================================

print("\n1. Loading training data...")

# Load feature-engineered training data
train_df = pd.read_csv('Data/features/spain_features_train.csv', index_col=0, parse_dates=True)

# Define target columns (24-hour multi-output)
CONSUMPTION_TARGETS = [f'target_consumption_h{i}' for i in range(1, 25)]
PRICE_TARGETS = [f'target_price_h{i}' for i in range(1, 25)]
ALL_TARGET_COLS = CONSUMPTION_TARGETS + PRICE_TARGETS

# Extract feature columns (exclude all targets)
feature_cols = [col for col in train_df.columns if col not in ALL_TARGET_COLS]

# Extract X and y (price only)
X_train = train_df[feature_cols].copy()
y_train_price = train_df[PRICE_TARGETS].values  # shape: (n_samples, 24)

print(f"  X_train shape: {X_train.shape}")
print(f"  y_train_price shape: {y_train_price.shape}")
print(f"  Feature columns: {len(feature_cols)}")

# ============================================================================
# 2. PREPROCESSING (same as original training)
# ============================================================================

print("\n2. Preprocessing (imputation + scaling)...")

# Handle infinity values
X_train = X_train.replace([np.inf, -np.inf], np.nan)
print(f"  Infinity values replaced with NaN")

# Imputation
imputer = SimpleImputer(strategy='median')
X_train_imputed = imputer.fit_transform(X_train)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)

print(f"  Imputation: Complete (median strategy)")
print(f"  Scaling: Applied (StandardScaler)")
print(f"  After preprocessing: X shape {X_train_scaled.shape}")
print(f"  No NaN values: {not np.isnan(X_train_scaled).any()}")

# ============================================================================
# 3. TRAIN XGBoost WITH QUANTILE LOSS
# ============================================================================

print("\n3. Training XGBoost with quantile loss (tau=0.90)...")

# Original hyperparameters (preserved)
xgb_params = {
    'n_estimators': 500,
    'learning_rate': 0.05,
    'max_depth': 7,
    'min_child_weight': 3,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'objective': 'reg:quantileerror',  # Quantile regression
    'quantile_alpha': 0.90,  # Focus on 90th percentile (top 10%)
    'random_state': 42,
    'n_jobs': -1,
    'verbosity': 0,
}

print(f"  Hyperparameters:")
for key, value in xgb_params.items():
    if key not in ['n_jobs', 'random_state', 'verbosity']:
        print(f"    {key}: {value}")

# Train with MultiOutputRegressor for 24-hour predictions
xgb_quantile = MultiOutputRegressor(
    xgb.XGBRegressor(**xgb_params),
    n_jobs=1  # Sequential training for 24 outputs
)

xgb_quantile.fit(X_train_scaled, y_train_price)
print("  Training complete!")

# ============================================================================
# 4. SAVE MODEL AND PREPROCESSING ARTIFACTS
# ============================================================================

print("\n4. Saving models to outputs/models_quantile/...")

output_dir = Path("outputs/models_quantile")
output_dir.mkdir(parents=True, exist_ok=True)

# Save the quantile model
model_path = output_dir / "xgb_price_quantile.pkl"
with open(model_path, 'wb') as f:
    pickle.dump(xgb_quantile, f)
print(f"  ✓ Saved: xgb_price_quantile.pkl ({model_path.stat().st_size / 1e6:.2f} MB)")

# Save preprocessing artifacts (for consistency with validation)
imputer_path = output_dir / "imputer_quantile.pkl"
with open(imputer_path, 'wb') as f:
    pickle.dump(imputer, f)
print(f"  ✓ Saved: imputer_quantile.pkl")

scaler_path = output_dir / "scaler_xgb_quantile.pkl"
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)
print(f"  ✓ Saved: scaler_xgb_quantile.pkl")

# ============================================================================
# 5. QUICK VALIDATION ON TRAINING DATA
# ============================================================================

print("\n5. Quick validation on training data...")

y_pred = xgb_quantile.predict(X_train_scaled)
print(f"  Prediction shape: {y_pred.shape}")

# Calculate training RMSE
train_rmse = np.sqrt(np.mean((y_train_price - y_pred) ** 2))
train_mae = np.mean(np.abs(y_train_price - y_pred))

print(f"  Training RMSE (quantile): {train_rmse:.2f}")
print(f"  Training MAE (quantile): {train_mae:.2f}")
print(f"  Baseline RMSE: 22.90")
print(f"  Expected: Some RMSE increase is normal with quantile loss")

# ============================================================================
# SUMMARY
# ============================================================================

summary = f"""
PHASE 2a COMPLETE: XGBoost Quantile Training
=============================================
✓ Training data loaded and preprocessed
✓ XGBoost trained with quantile loss (tau=0.90)
✓ 24 estimators (one per hour) trained
✓ Model saved to: outputs/models_quantile/xgb_price_quantile.pkl
✓ Preprocessing artifacts saved for validation

Model Configuration:
  - Objective: reg:quantilehuber (quantile regression)
  - Quantile alpha: 0.90 (90th percentile)
  - N_estimators: 500 per hour
  - Max_depth: 7 (preserved from original)

Next Step: Run Phase 2b and 2c retraining scripts
Then: Execute Phase 3 validation comparison
"""

print("\n" + "=" * 80)
print(summary)
print("=" * 80)
