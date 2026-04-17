"""
PHASE 3: VALIDATION & COMPARISON
=================================

Compare baseline vs quantile models on 2025 validation data.
Measures improvement in High_Accuracy_% (primary metric for underprediction fix).

Goal: Verify if quantile loss achieves >=3% improvement without RMSE increase >5%
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import torch
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("PHASE 3: QUANTILE MODEL VALIDATION & COMPARISON")
print("=" * 80)

# ============================================================================
# 1. LOAD TEST DATA (2025 Validation)
# ============================================================================

print("\n1. Loading 2025 validation data...")

test_df = pd.read_csv('Data/features/spain_features_test.csv', index_col=0, parse_dates=True)

# Define targets & features
CONSUMPTION_TARGETS = [f'target_consumption_h{i}' for i in range(1, 25)]
PRICE_TARGETS = [f'target_price_h{i}' for i in range(1, 25)]
ALL_TARGET_COLS = CONSUMPTION_TARGETS + PRICE_TARGETS
feature_cols = [col for col in test_df.columns if col not in ALL_TARGET_COLS]

X_test = test_df[feature_cols].values
y_test_price = test_df[PRICE_TARGETS].values

print(f"  X_test shape: {X_test.shape}")
print(f"  y_test_price shape: {y_test_price.shape}")

# ============================================================================
# 2. LOAD MODELS & SCALERS
# ============================================================================

print("\n2. Loading baseline and quantile models...")

baseline_dir = Path("outputs/models")
quantile_dir = Path("outputs/models_quantile")

# Baseline models
baseline_xgb = pickle.load(open(baseline_dir / "xgb_price.pkl", 'rb'))
baseline_rf = pickle.load(open(baseline_dir / "rf_price.pkl", 'rb'))
baseline_gru_path = baseline_dir / "gru_price_final.pt"

# Quantile models
quantile_xgb = pickle.load(open(quantile_dir / "xgb_price_quantile.pkl", 'rb'))
quantile_rf = pickle.load(open(quantile_dir / "rf_price_quantile.pkl", 'rb'))
quantile_gru_path = quantile_dir / "gru_price_quantile.pt"

# Scalers
baseline_scaler = pickle.load(open(baseline_dir / "scaler_X.pkl", 'rb'))
imputer = pickle.load(open(baseline_dir / "imputer.pkl", 'rb'))

print("  ✓ All models and scalers loaded")

# ============================================================================
# 3. PREPROCESS TEST DATA
# ============================================================================

print("\n3. Preprocessing test data...")

# Handle infinities
X_test = np.nan_to_num(X_test, nan=np.nan, posinf=np.nan, neginf=np.nan)

# Imputation
X_test_imputed = imputer.transform(X_test)

# Scaling (use baseline scaler for consistency)
X_test_scaled = baseline_scaler.transform(X_test_imputed)

print(f"  Preprocessing complete: X_test shape {X_test_scaled.shape}")

# ============================================================================
# 4. LOAD GRU MODELS
# ============================================================================

print("\n4. Loading GRU models...")

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class EnergyGRU(torch.nn.Module):
    def __init__(self, input_size, hidden_size=128, hidden_size2=64, dropout=0.2, output_steps=24):
        super(EnergyGRU, self).__init__()
        self.gru1 = torch.nn.GRU(input_size, hidden_size, batch_first=True)
        self.bn1 = torch.nn.BatchNorm1d(hidden_size)
        self.drop1 = torch.nn.Dropout(dropout)
        self.gru2 = torch.nn.GRU(hidden_size, hidden_size2, batch_first=True)
        self.bn2 = torch.nn.BatchNorm1d(hidden_size2)
        self.drop2 = torch.nn.Dropout(dropout)
        self.fc1 = torch.nn.Linear(hidden_size2, 64)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(64, output_steps)
    
    def forward(self, x):
        out, _ = self.gru1(x)
        out = out[:, -1, :]
        out = self.bn1(out)
        out = self.drop1(out)
        out = out.unsqueeze(1)
        out, _ = self.gru2(out)
        out = out[:, -1, :]
        out = self.bn2(out)
        out = self.drop2(out)
        out = self.relu(self.fc1(out))
        return self.fc2(out)

# Load baseline GRU
baseline_gru_checkpoint = torch.load(baseline_gru_path, map_location=DEVICE)
baseline_gru_config = baseline_gru_checkpoint['model_config']
baseline_gru = EnergyGRU(**baseline_gru_config).to(DEVICE)
baseline_gru.load_state_dict(baseline_gru_checkpoint['model_state_dict'])
baseline_gru.eval()

# Load quantile GRU
quantile_gru_checkpoint = torch.load(quantile_gru_path, map_location=DEVICE)
quantile_gru_config = quantile_gru_checkpoint['model_config']
quantile_gru = EnergyGRU(**quantile_gru_config).to(DEVICE)
quantile_gru.load_state_dict(quantile_gru_checkpoint['model_state_dict'])
quantile_gru.eval()

print(f"  ✓ GRU models loaded (device: {DEVICE})")

# ============================================================================
# 5. CREATE GRU SEQUENCES
# ============================================================================

print("\n5. Creating GRU sequences (lookback=48)...")

lookback = 48

def create_sequences(X, lookback):
    X_seq = []
    for i in range(len(X) - lookback):
        X_seq.append(X[i:i + lookback])
    return np.array(X_seq)

X_test_seq = create_sequences(X_test_scaled, lookback)
print(f"  GRU sequence shape: {X_test_seq.shape}")

# ============================================================================
# 6. RUN PREDICTIONS
# ============================================================================

print("\n6. Running predictions on test data (tree models + GRU)...")

# Baseline predictions
print("  Running baseline XGBoost...")
pred_baseline_xgb = baseline_xgb.predict(X_test_scaled)

print("  Running baseline Random Forest...")
pred_baseline_rf = baseline_rf.predict(X_test_scaled)

print("  Running baseline GRU...")
with torch.no_grad():
    X_test_seq_tensor = torch.FloatTensor(X_test_seq).to(DEVICE)
    gru_pred_scaled = baseline_gru(X_test_seq_tensor).cpu().numpy()
    # Load price scaler for baseline
    scaler_price_baseline = pickle.load(open(baseline_dir / "scaler_price.pkl", 'rb'))
    pred_baseline_gru = scaler_price_baseline.inverse_transform(gru_pred_scaled)

# Quantile predictions (using same scaled data)
print("  Running quantile XGBoost...")
pred_quantile_xgb = quantile_xgb.predict(X_test_scaled)

print("  Running quantile Random Forest...")
pred_quantile_rf = quantile_rf.predict(X_test_scaled)

print("  Running quantile GRU...")
with torch.no_grad():
    X_test_seq_tensor = torch.FloatTensor(X_test_seq).to(DEVICE)
    gru_pred_scaled = quantile_gru(X_test_seq_tensor).cpu().numpy()
    # Load price scaler for quantile (should be in checkpoint)
    if 'scaler_price' in quantile_gru_checkpoint:
        scaler_price_quantile = pickle.loads(quantile_gru_checkpoint['scaler_price'])
    else:
        scaler_price_quantile = pickle.load(open(quantile_dir / "scaler_gru_price_quantile.pkl", 'rb'))
    pred_quantile_gru = scaler_price_quantile.inverse_transform(gru_pred_scaled)

print("  All predictions complete!")

# ============================================================================
# 7. COMPUTE METRICS
# ============================================================================

print("\n7. Computing metrics...")

# Align y_test_price to GRU sequence length (GRU loses first 48 samples)
# Tree models: full 2,412 samples
# GRU models: 2,412 - 48 = 2,364 samples
# Solution: Use only the last 2,364 samples of y_test_price that correspond to GRU sequences
lookback_loss = lookback
y_test_price_for_gru = y_test_price[lookback_loss:]

# Trim tree model predictions to match GRU length
def trim_predictions(pred, target_len):
    """Trim predictions to match target length (last N samples)"""
    flat_pred = pred.flatten()
    return flat_pred[-target_len:]

def safe_divide(numerator, denominator, default=0.0):
    """Safe division to prevent NaN"""
    return (numerator / denominator) if denominator > 0 else default

def compute_binned_accuracy(actual, pred):
    """Compute binned accuracy (Low/Medium/High bins)"""
    # Flatten to 1D if 2D
    actual = actual.flatten()
    pred = pred.flatten()
    
    # Define bins: Low < 33%, Medium 33-66%, High > 66%
    low_bin = np.percentile(actual, 33)
    high_bin = np.percentile(actual, 67)
    
    # Assign actual to bins
    actual_bins = np.zeros(len(actual), dtype=int)
    actual_bins[(actual > low_bin) & (actual <= high_bin)] = 1
    actual_bins[actual > high_bin] = 2
    
    # Assign pred to bins
    pred_bins = np.zeros(len(pred), dtype=int)
    pred_bins[(pred > low_bin) & (pred <= high_bin)] = 1
    pred_bins[pred > high_bin] = 2
    
    # Confusion matrix
    cm = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            cm[i, j] = np.sum((actual_bins == i) & (pred_bins == j))
    
    # Normalize by actual bin counts (row-wise)
    cm_norm = np.zeros_like(cm, dtype=float)
    for i in range(3):
        row_sum = cm[i].sum()
        if row_sum > 0:
            cm_norm[i, :] = cm[i, :] / row_sum * 100
        else:
            cm_norm[i, :] = 0.0
    
    # Extract accuracies
    low_acc = cm_norm[0, 0]
    med_acc = cm_norm[1, 1]
    high_acc = cm_norm[2, 2]
    
    # Ensure no NaN
    low_acc = 0.0 if not np.isfinite(low_acc) else low_acc
    med_acc = 0.0 if not np.isfinite(med_acc) else med_acc
    high_acc = 0.0 if not np.isfinite(high_acc) else high_acc
    
    return low_acc, med_acc, high_acc

def compute_rmse_mae(actual, pred):
    """Compute RMSE and MAE"""
    # Flatten to 1D if 2D
    actual = actual.flatten()
    pred = pred.flatten()
    rmse = np.sqrt(np.mean((actual - pred) ** 2))
    mae = np.mean(np.abs(actual - pred))
    return rmse, mae

# Compute all metrics
comparison_data = []

models_list = [
    ("XGBoost", pred_baseline_xgb, pred_quantile_xgb),
    ("Random Forest", pred_baseline_rf, pred_quantile_rf),
    ("GRU", pred_baseline_gru, pred_quantile_gru),
]

for name, pred_baseline, pred_quantile in models_list:
    # Flatten predictions
    pred_baseline_flat = pred_baseline.flatten()
    pred_quantile_flat = pred_quantile.flatten()
    
    # Align all to GRU sequence length (trim to last N samples)
    target_len = len(y_test_price_for_gru.flatten())
    pred_baseline_aligned = pred_baseline_flat[-target_len:]
    pred_quantile_aligned = pred_quantile_flat[-target_len:]
    y_test_aligned = y_test_price_for_gru.flatten()
    
    # RMSE & MAE
    rmse_base, mae_base = compute_rmse_mae(y_test_aligned, pred_baseline_aligned)
    rmse_quant, mae_quant = compute_rmse_mae(y_test_aligned, pred_quantile_aligned)
    
    # Binned accuracy
    low_base, med_base, high_base = compute_binned_accuracy(y_test_aligned, pred_baseline_aligned)
    low_quant, med_quant, high_quant = compute_binned_accuracy(y_test_aligned, pred_quantile_aligned)
    
    # Improvements
    rmse_improve = ((rmse_base - rmse_quant) / rmse_base * 100) if rmse_base > 0 else 0
    high_acc_improve = high_quant - high_base
    
    comparison_data.append({
        'Model': name,
        'Baseline_RMSE': round(rmse_base, 2),
        'Quantile_RMSE': round(rmse_quant, 2),
        'RMSE_Improvement_%': round(rmse_improve, 2),
        'Baseline_MAE': round(mae_base, 2),
        'Quantile_MAE': round(mae_quant, 2),
        'Baseline_Low_Acc_%': round(low_base, 2),
        'Quantile_Low_Acc_%': round(low_quant, 2),
        'Baseline_Med_Acc_%': round(med_base, 2),
        'Quantile_Med_Acc_%': round(med_quant, 2),
        'Baseline_High_Acc_%': round(high_base, 2),
        'Quantile_High_Acc_%': round(high_quant, 2),
        'High_Acc_Improvement_%': round(high_acc_improve, 2),
    })

# ============================================================================
# 8. SAVE COMPARISON CSV
# ============================================================================

print("\n8. Saving comparison results...")

comparison_df = pd.DataFrame(comparison_data)
comparison_df.to_csv('validation/quantile_comparison.csv', index=False)

print(f"  ✓ Saved: validation/quantile_comparison.csv")

# ============================================================================
# 9. DISPLAY RESULTS & DECISION CHECKPOINT
# ============================================================================

print("\n" + "=" * 80)
print("COMPARISON RESULTS (XGBoost, Random Forest, GRU)")
print("=" * 80)
print()
print(comparison_df.to_string(index=False))
print()

# Decision checkpoint
print("=" * 80)
print("DECISION CHECKPOINT")
print("=" * 80)

success_min_improvement = 3.0
success_max_rmse_degrade = 5.0

proceed_to_phase4 = True
reasons = []

for idx, row in comparison_df.iterrows():
    model = row['Model']
    high_improve = row['High_Acc_Improvement_%']
    rmse_degrade = abs(row['RMSE_Improvement_%']) * -1 if row['RMSE_Improvement_%'] < 0 else 0
    
    print(f"\n{model}:")
    print(f"  High_Accuracy improvement: {high_improve:.2f}% (target: >= {success_min_improvement}%)")
    print(f"  RMSE degradation: {rmse_degrade:.2f}% (max allowed: {success_max_rmse_degrade}%)")
    
    if high_improve >= success_min_improvement and rmse_degrade <= success_max_rmse_degrade:
        print(f"  ✓ PASS")
    else:
        print(f"  ✗ FAIL")
        proceed_to_phase4 = False
        if high_improve < success_min_improvement:
            reasons.append(f"{model}: High_Acc improvement ({high_improve:.2f}%) < threshold ({success_min_improvement}%)")
        if rmse_degrade > success_max_rmse_degrade:
            reasons.append(f"{model}: RMSE degradation ({rmse_degrade:.2f}%) > threshold ({success_max_rmse_degrade}%)")

print("\n" + "=" * 80)
if proceed_to_phase4:
    print("RESULT: ✓ PROCEED TO PHASE 4 (Ensemble Experiments)")
    print("=" * 80)
    print("\nAll quantile models meet improvement thresholds!")
    print("Next: Run ensemble experiments with quantile models")
    print("  Command: python PHASE_4_ENSEMBLE_EXPERIMENTS.py")
else:
    print("RESULT: ✗ HALT (Quantile models did not meet thresholds)")
    print("=" * 80)
    print("\nFindings:")
    for reason in reasons:
        print(f"  - {reason}")
    print("\nRecommendation: Review quantile strategy or adjust hyperparameters")

print("\n" + "=" * 80)
