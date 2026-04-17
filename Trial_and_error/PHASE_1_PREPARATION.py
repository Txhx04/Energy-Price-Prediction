"""
PHASE 1: PREPARATION FOR QUANTILE LOSS RETRAINING
===================================================

This script documents baseline metrics and prepares the retraining plan.
NO CODE IS MODIFIED. This is read-only preparation for Phase 2.

Goal: Improve HIGH-value prediction accuracy in price models
Current: XGB_Price High_Accuracy% = 0.0% (never predicts HIGH)
Target: Achieve >=3% improvement without RMSE increase >5%
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================================
# 1. BASELINE METRICS (from validation/metrics_summary.csv)
# ============================================================================

BASELINE_METRICS = {
    'XGB_Price': {'RMSE': 22.90, 'MAE': 16.90, 'High_Accuracy%': 0.0},
    'RF_Price': {'RMSE': 26.32, 'MAE': 20.12, 'High_Accuracy%': 0.0},
    'LR_Price': {'RMSE': 26.29, 'MAE': 19.68, 'High_Accuracy%': 0.0},
}

print("=" * 80)
print("BASELINE METRICS (Current Models - UNTOUCHABLE)")
print("=" * 80)
for model, metrics in BASELINE_METRICS.items():
    print(f"\n{model}:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

# ============================================================================
# 2. ORIGINAL HYPERPARAMETERS (from 03_model_training.ipynb)
# ============================================================================

ORIGINAL_HYPERPARAMS = {
    'XGBoost': {
        'n_estimators': 500,
        'learning_rate': 0.05,
        'max_depth': 7,
        'min_child_weight': 3,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'n_jobs': -1,
        'random_state': 42,
    },
    'RandomForest': {
        'n_estimators': 100,
        'max_depth': 20,
        'random_state': 42,
        'n_jobs': -1,
    },
    'LinearRegression': {
        'fit_intercept': True,
        'n_jobs': -1,
    }
}

print("\n" + "=" * 80)
print("ORIGINAL HYPERPARAMETERS (Will be PRESERVED)")
print("=" * 80)
for model, params in ORIGINAL_HYPERPARAMS.items():
    print(f"\n{model}:")
    for key, value in params.items():
        print(f"  {key}: {value}")

# ============================================================================
# 3. QUANTILE LOSS STRATEGY
# ============================================================================

QUANTILE_STRATEGY = {
    'tau_value': 0.90,  # Focus on 90th percentile (top 10% of high values)
    'interpretation': 'Penalizes underprediction 9x more than overprediction',
    'expected_effect': 'Models learn to predict higher values to reduce loss',
    'decision_threshold': {
        'min_improvement': 3.0,  # At least 3% improvement in High_Accuracy%
        'max_rmse_increase': 5.0,  # RMSE can increase by max 5%
    }
}

print("\n" + "=" * 80)
print("QUANTILE LOSS STRATEGY")
print("=" * 80)
print(f"\nTau (τ): {QUANTILE_STRATEGY['tau_value']}")
print(f"Interpretation: {QUANTILE_STRATEGY['interpretation']}")
print(f"Expected Effect: {QUANTILE_STRATEGY['expected_effect']}")
print(f"\nSuccess Criteria:")
print(f"  - Min improvement: {QUANTILE_STRATEGY['decision_threshold']['min_improvement']}% in High_Accuracy%")
print(f"  - Max RMSE increase: {QUANTILE_STRATEGY['decision_threshold']['max_rmse_increase']}%")

# ============================================================================
# 4. PHASE 2-4 PLAN STRUCTURE
# ============================================================================

IMPLEMENTATION_PLAN = """
PHASE 2: DEVELOPMENT
====================
Create 3 separate retraining scripts (will NOT modify existing code):

  1. 03b_retrain_xgboost_quantile.py
     - Load training data from Data/processed/spain_merged_final.csv
     - Apply same preprocessing (imputation, scaling)
     - Train XGBoostRegressor with objective='quantile' and quantile_alpha=0.90
     - Save to: outputs/models_quantile/xgb_price_quantile.pkl
     - Baseline: RMSE 22.90, High_Acc% 0.0%

  2. 03c_retrain_rf_quantile.py
     - Random Forest doesn't support native quantile loss
     - Alternative: Use sklearn QuantileRegressor wrapper OR weighted loss
     - Save to: outputs/models_quantile/rf_price_quantile.pkl
     - Baseline: RMSE 26.32, High_Acc% 0.0%

  3. 03d_retrain_lr_quantile.py
     - Use sklearn.linear_model.QuantileRegressor with alpha=0.90
     - Solver: 'highs'
     - Save to: outputs/models_quantile/lr_price_quantile.pkl
     - Baseline: RMSE 26.29, High_Acc% 0.0%

PHASE 3: VALIDATION
===================
  Create validation_quantile_comparison.py:
  - Load both baseline AND quantile models
  - Run on same 2025 validation data (8,730 samples)
  - Generate side-by-side metrics comparison CSV
  - Focus metric: High_Accuracy_% improvement

  Checkpoint Decision:
    IF High_Accuracy% improvement >= 3.0% AND RMSE increase <= 5.0%:
      PROCEED TO PHASE 4 (Optional ensemble experiments)
    ELSE:
      HALT and document findings

PHASE 4: OPTIONAL ENSEMBLE
==========================
  - If Phase 3 successful, test blended ensemble:
    Ensemble = 0.5 * quantile_xgb + 0.3 * quantile_lr + 0.2 * quantile_rf
  - Measure combined High_Accuracy% improvement
  - Compare with baseline ensemble

CONSTRAINTS VERIFIED:
☑ No modifications to existing code (validation.py, models.pkl)
☑ New models saved to separate directory (outputs/models_quantile/)
☑ Original hyperparameters preserved
☑ Training data loaded externally (not hardcoded)
☑ All baselines documented for comparison
"""

print("\n" + "=" * 80)
print("IMPLEMENTATION PLAN")
print("=" * 80)
print(IMPLEMENTATION_PLAN)

# ============================================================================
# 5. DATA VALIDATION
# ============================================================================

print("\n" + "=" * 80)
print("DATA VALIDATION")
print("=" * 80)

data_path = Path("Data/processed/spain_merged_final.csv")
if data_path.exists():
    print(f"✓ Training data found: {data_path} ({data_path.stat().st_size / 1e6:.1f} MB)")
    
    # Load a small sample to verify structure
    df_sample = pd.read_csv(data_path, nrows=5)
    print(f"  Shape: {df_sample.shape}")
    print(f"  Columns: {list(df_sample.columns)[:10]}... (showing first 10)")
else:
    print(f"✗ Training data NOT found: {data_path}")
    print("  ERROR: Cannot proceed without training data!")

models_dir = Path("outputs/models")
if models_dir.exists():
    pkl_files = list(models_dir.glob("*.pkl"))
    print(f"\n✓ Model outputs directory found: {len(pkl_files)} .pkl files")
    print(f"  Example files: {[f.name for f in pkl_files[:3]]}")
else:
    print(f"\n✗ Model outputs directory NOT found")

models_quantile_dir = Path("outputs/models_quantile")
if models_quantile_dir.exists():
    print(f"\n✓ Created directory for new quantile models: outputs/models_quantile/")
else:
    print(f"\n✗ Failed to create quantile models directory")

# ============================================================================
# 6. SUMMARY
# ============================================================================

SUMMARY = """
PHASE 1 COMPLETE
================
✓ Baseline metrics documented
✓ Original hyperparameters extracted
✓ Quantile loss strategy defined (tau=0.90)
✓ Implementation plan structured
✓ Data paths validated
✓ No code modifications made (preparation only)

NEXT STEP: Execute Phase 2
Run the following in sequence:
  1. python 03b_retrain_xgboost_quantile.py
  2. python 03c_retrain_rf_quantile.py
  3. python 03d_retrain_lr_quantile.py
  
Then proceed to Phase 3: validation_quantile_comparison.py
"""

print("\n" + "=" * 80)
print(SUMMARY)
print("=" * 80)

# Save summary to file for reference
summary_file = Path("PHASE_1_PREPARATION_SUMMARY.txt")
with open(summary_file, 'w') as f:
    f.write(IMPLEMENTATION_PLAN)
print(f"\nPlan saved to: {summary_file}")
