"""
PHASE 2d: RETRAIN GRU WITH QUANTILE-INSPIRED APPROACH
======================================================

Trains GRU price prediction model similar to XGBoost quantile retraining.
Uses weighted loss to emphasize high-value predictions (quantile-inspired).

Saves to: outputs/models_quantile/gru_price_quantile.pt

GPU Required: Yes (CUDA/MPS)
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("PHASE 2d: GRU Quantile-Inspired Price Retraining")
print("=" * 80)

# ============================================================================
# 1. DEVICE SETUP
# ============================================================================

print("\n1. Setting up GPU...")

if torch.cuda.is_available():
    DEVICE = torch.device('cuda:0')
    print(f"  ✓ GPU available: {torch.cuda.get_device_name(0)}")
    print(f"    CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
    print(f"  ✓ MPS (Metal Performance Shaders) available")
else:
    DEVICE = torch.device('cpu')
    print(f"  ⚠ GPU NOT available, using CPU (this will be slow)")

# ============================================================================
# 2. LOAD AND PREPROCESS TRAINING DATA
# ============================================================================

print("\n2. Loading training data...")

train_df = pd.read_csv('Data/features/spain_features_train.csv', index_col=0, parse_dates=True)

# Define targets & features
CONSUMPTION_TARGETS = [f'target_consumption_h{i}' for i in range(1, 25)]
PRICE_TARGETS = [f'target_price_h{i}' for i in range(1, 25)]
ALL_TARGET_COLS = CONSUMPTION_TARGETS + PRICE_TARGETS

feature_cols = [col for col in train_df.columns if col not in ALL_TARGET_COLS]

X_train = train_df[feature_cols].values
y_train_price = train_df[PRICE_TARGETS].values

print(f"  X_train shape: {X_train.shape}")
print(f"  y_train_price shape: {y_train_price.shape}")

# ============================================================================
# 3. PREPROCESSING
# ============================================================================

print("\n3. Preprocessing data...")

# Handle infinities
X_train = np.nan_to_num(X_train, nan=np.nan, posinf=np.nan, neginf=np.nan)

# Imputation
imputer = SimpleImputer(strategy='median')
X_train_imputed = imputer.fit_transform(X_train)

# Scaling
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train_imputed)

# Scale targets for neural network training
scaler_price = StandardScaler()
y_train_scaled = scaler_price.fit_transform(y_train_price)

print(f"  X_train scaled shape: {X_train_scaled.shape}")
print(f"  y_train_scaled shape: {y_train_scaled.shape}")
print(f"  No NaN values: {not np.isnan(X_train_scaled).any()}")

# ============================================================================
# 4. CREATE SEQUENCES FOR GRU
# ============================================================================

print("\n4. Creating sequences for GRU (lookback=48)...")

lookback = 48

def create_sequences(X, y, lookback):
    """Create sequences for RNN training"""
    X_seq = []
    y_seq = []
    
    for i in range(len(X) - lookback):
        seq = X[i:i + lookback]
        X_seq.append(seq)
        y_seq.append(y[i + lookback])
    
    return np.array(X_seq), np.array(y_seq)

X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, lookback)

print(f"  X_train_seq shape: {X_train_seq.shape}")  # (n_samples, 48, 82)
print(f"  y_train_seq shape: {y_train_seq.shape}")  # (n_samples, 24)

# ============================================================================
# 5. CREATE DATA LOADERS
# ============================================================================

print("\n5. Creating data loaders...")

batch_size = 64
train_dataset = TensorDataset(
    torch.FloatTensor(X_train_seq),
    torch.FloatTensor(y_train_seq)
)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

print(f"  Batch size: {batch_size}")
print(f"  Total batches: {len(train_loader)}")

# ============================================================================
# 6. GRU ARCHITECTURE
# ============================================================================

print("\n6. Defining GRU architecture...")

class EnergyGRU(nn.Module):
    """GRU for 24-hour energy price forecasting"""
    def __init__(self, input_size, hidden_size=128, hidden_size2=64, 
                 dropout=0.2, output_steps=24):
        super(EnergyGRU, self).__init__()
        
        self.gru1   = nn.GRU(input_size, hidden_size, batch_first=True)
        self.bn1    = nn.BatchNorm1d(hidden_size)
        self.drop1  = nn.Dropout(dropout)
        
        self.gru2   = nn.GRU(hidden_size, hidden_size2, batch_first=True)
        self.bn2    = nn.BatchNorm1d(hidden_size2)
        self.drop2  = nn.Dropout(dropout)
        
        self.fc1    = nn.Linear(hidden_size2, 64)
        self.relu   = nn.ReLU()
        self.fc2    = nn.Linear(64, output_steps)
    
    def forward(self, x):
        # First GRU layer
        out, _ = self.gru1(x)
        out = out[:, -1, :]  # Take last timestep
        out = self.bn1(out)
        out = self.drop1(out)
        
        # Second GRU layer
        out = out.unsqueeze(1)  # Add sequence dimension
        out, _ = self.gru2(out)
        out = out[:, -1, :]  # Take last timestep
        out = self.bn2(out)
        out = self.drop2(out)
        
        # Fully connected layers
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        
        return out

input_size = X_train_seq.shape[2]  # 82
gru_model = EnergyGRU(input_size=input_size, output_steps=24).to(DEVICE)

print(f"  Model architecture defined")
print(f"  Input size: {input_size}")
print(f"  Output steps: 24")
print(f"  Total parameters: {sum(p.numel() for p in gru_model.parameters()):,}")

# ============================================================================
# 7. TRAINING WITH WEIGHTED LOSS (QUANTILE-INSPIRED)
# ============================================================================

print("\n7. Training GRU with weighted loss (quantile-inspired)...")

# Weight samples: higher weight for high-price samples
sample_weights = 1.0 / (np.mean(y_train_price, axis=1) + 1e-6)
sample_weights = sample_weights[lookback:]  # Align with sequences
sample_weights = sample_weights / np.mean(sample_weights)  # Normalize
sample_weights_tensor = torch.FloatTensor(sample_weights).to(DEVICE)

# Training hyperparameters
optimizer = torch.optim.Adam(gru_model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.5, patience=5, verbose=False, min_lr=1e-6
)
criterion = nn.MSELoss(reduction='none')

epochs = 50
patience = 10
best_val_loss = float('inf')
patience_counter = 0
best_weights = None

print(f"  Epochs: {epochs}, Patience: {patience}")
print(f"  Device: {DEVICE}")
print(f"  Learning rate: 0.001")
print()

print(f"{'Epoch':>6} | {'Train Loss':>12} | {'LR':>10} | Status")
print("-" * 50)

for epoch in range(1, epochs + 1):
    gru_model.train()
    
    train_loss_total = 0.0
    num_samples = 0
    
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)
        
        # Forward pass
        optimizer.zero_grad()
        predictions = gru_model(X_batch)
        
        # Weighted loss (higher weight for high-price samples)
        batch_weights = sample_weights_tensor[:X_batch.size(0)]
        loss_per_sample = criterion(predictions, y_batch).mean(dim=1)  # Mean across 24 hours
        weighted_loss = (loss_per_sample * batch_weights).mean()
        
        # Backward pass
        weighted_loss.backward()
        torch.nn.utils.clip_grad_norm_(gru_model.parameters(), 1.0)
        optimizer.step()
        
        train_loss_total += weighted_loss.item() * X_batch.size(0)
        num_samples += X_batch.size(0)
    
    train_loss_avg = train_loss_total / num_samples
    
    # Learning rate adjustment
    scheduler.step(train_loss_avg)
    lr_now = optimizer.param_groups[0]['lr']
    
    # Print progress
    if epoch % 5 == 0 or epoch == 1:
        print(f"{epoch:>6} | {train_loss_avg:>12.6f} | {lr_now:>10.6f} | ", end="")
        
        if train_loss_avg < best_val_loss:
            best_val_loss = train_loss_avg
            best_weights = {k: v.cpu().clone() for k, v in gru_model.state_dict().items()}
            patience_counter = 0
            print("✓ best")
        else:
            patience_counter += 1
            print(f"patience {patience_counter}/{patience}")
        
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break

# Load best weights
if best_weights is not None:
    gru_model.load_state_dict(best_weights)

print(f"\n✓ Training complete. Best loss: {best_val_loss:.6f}")

# ============================================================================
# 8. EVALUATE ON TRAINING DATA (Batch-wise to avoid GPU OOM)
# ============================================================================

print("\n8. Evaluating on training data (batch-wise)...")

gru_model.eval()
pred_train_scaled_list = []

with torch.no_grad():
    for X_batch, _ in train_loader:
        X_batch = X_batch.to(DEVICE)
        batch_pred = gru_model(X_batch).cpu().numpy()
        pred_train_scaled_list.append(batch_pred)

pred_train_scaled = np.vstack(pred_train_scaled_list)

# Inverse transform predictions
pred_train = scaler_price.inverse_transform(pred_train_scaled)
actual_train = y_train_price[lookback:]  # Align with sequences

train_rmse = np.sqrt(np.mean((actual_train - pred_train) ** 2))
train_mae = np.mean(np.abs(actual_train - pred_train))

print(f"  Training RMSE (weighted): {train_rmse:.2f}")
print(f"  Training MAE (weighted): {train_mae:.2f}")
print(f"  Baseline RMSE (from validation): 23.64")

# ============================================================================
# 9. SAVE MODEL
# ============================================================================

print("\n9. Saving model to outputs/models_quantile/...")

output_dir = Path("outputs/models_quantile")
output_dir.mkdir(parents=True, exist_ok=True)

model_path = output_dir / "gru_price_quantile.pt"
torch.save({
    'model_state_dict': gru_model.state_dict(),
    'model_config': {
        'input_size': input_size,
        'hidden_size': 128,
        'hidden_size2': 64,
        'dropout': 0.2,
        'output_steps': 24
    },
    'scaler_price': pickle.dumps(scaler_price),
    'scaler_X': pickle.dumps(scaler_X),
    'imputer': pickle.dumps(imputer),
}, model_path)

print(f"  ✓ Saved: gru_price_quantile.pt ({model_path.stat().st_size / 1e6:.2f} MB)")

# Save scalers separately for easy access
scaler_price_path = output_dir / "scaler_gru_price_quantile.pkl"
with open(scaler_price_path, 'wb') as f:
    pickle.dump(scaler_price, f)
print(f"  ✓ Saved: scaler_gru_price_quantile.pkl")

# ============================================================================
# SUMMARY
# ============================================================================

summary = f"""
PHASE 2d COMPLETE: GRU Quantile Price Training
===============================================
✓ Training data loaded and preprocessed
✓ Sequences created (lookback=48)
✓ GRU trained with weighted loss (sample weights emphasize high prices)
✓ GPU used for training: {DEVICE}
✓ Model saved to: outputs/models_quantile/gru_price_quantile.pt
✓ Scalers saved for validation

Model Configuration:
  - Input size: {input_size}
  - Hidden size: 128
  - Hidden size 2: 64
  - Dropout: 0.2
  - Output steps: 24 (hours)
  - Lookback: 48 (hours)

Training Details:
  - Batch size: {batch_size}
  - Optimizer: Adam
  - Learning rate: 0.001 (with ReduceLROnPlateau)
  - Loss: MSE with sample weights
  - Epochs: {epoch}
  - Best train loss: {best_val_loss:.6f}

Performance:
  - Training RMSE: {train_rmse:.2f}
  - Training MAE: {train_mae:.2f}
  - Baseline RMSE: 23.64

Next Step: Run Phase 3 validation comparison
  Command: python validation_quantile_comparison.py
  (Update it to include GRU quantile model)

Next Step (if desired): Add GRU quantile to ensemble in Phase 4
"""

print("\n" + "=" * 80)
print(summary)
print("=" * 80)
