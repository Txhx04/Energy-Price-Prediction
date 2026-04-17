"""
================================================================================
ENSEMBLE MODEL CREATION & MANAGEMENT
================================================================================
Ensemble voting mechanism combining XGBoost, GRU, RF predictions for robust forecasting.
Creates weighted ensemble models for consumption and price predictions.

Ensemble Configurations:
  CONSUMPTION: 60% XGBoost + 30% GRU + 10% Random Forest
  PRICE:       50% XGBoost + 35% GRU + 15% Random Forest

The ensemble combines the strengths of:
  - XGBoost: Gradient boosting and tree ensemble variance reduction logic capturing non-linear feature interactions.
  - GRU: Recurrent gate mechanism (GRU reset/update gates) for temporal dependency modeling.
  - Random Forest: Tree ensemble bagging for variance reduction and interpretability.
"""

import pickle
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler


class EnergyGRU(nn.Module):
    """
    GRU Architecture matching the trained model.
    Utilizes recurrent gate mechanisms (reset/update gates) for temporal dependency modeling
    and time-series sequence prediction.
    """
    def __init__(self, input_size, hidden_size=128, hidden_size2=64, dropout=0.2, output_steps=24):
        """
        Initialize GRU structure.
        
        Inputs:
          - input_size: int, number of features in input sequence
          - hidden_size: int, hidden state dimensions for first GRU layer
          - hidden_size2: int, hidden state dimensions for second GRU layer
          - dropout: float, dropout probability for regularization
          - output_steps: int, target sequence length to predict (default 24h)
          
        ML Principle: GRU hidden state initialization and reset gate logic to mitigate vanishing gradients.
        """
        super(EnergyGRU, self).__init__()
        self.gru1 = nn.GRU(input_size, hidden_size, batch_first=True)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.drop1 = nn.Dropout(dropout)
        self.gru2 = nn.GRU(hidden_size, hidden_size2, batch_first=True)
        self.bn2 = nn.BatchNorm1d(hidden_size2)
        self.drop2 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size2, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, output_steps)

    def forward(self, x):
        """
        Model forward pass through GRU cells.
        
        Inputs: 
          - x: torch.Tensor of shape [batch_size, seq_length, n_features]
          
        Outputs: 
          - predicted sequence: torch.Tensor of shape [batch_size, output_steps]
          
        ML Principle: Sequential processing through GRU reset/update gates, tracking hidden states.
        """
        out, _ = self.gru1(x)
        out = out[:, -1, :]
        # Batch normalization and standardization for numerical stability
        out = self.bn1(out)
        out = self.drop1(out)
        out = out.unsqueeze(1)
        out, _ = self.gru2(out)
        out = out[:, -1, :]
        out = self.bn2(out)
        out = self.drop2(out)
        out = self.relu(self.fc1(out))
        return self.fc2(out)


class EnsembleModel:
    """
    Ensemble predictor that combines XGBoost, GRU, and Random Forest models via weighted averaging.
    Applies ensemble aggregation logic (voting weights) to multiple model outputs to increase robustness.
    """
    
    def __init__(self, target_type='consumption', device='cuda'):
        """
        Initialize the Ensemble meta-model.
        
        Inputs:
          - target_type: str, 'consumption' or 'price', determines voting weights constraints.
          - device: str, 'cuda' or 'cpu' for GRU execution.
          
        Outputs: 
          - Mutates object state initializing weights correctly.
          
        ML Principle: Ensemble voting mechanism assigning weights by respective validation accuracy.
        """
        self.target_type = target_type
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Ensemble weights
        if target_type == 'consumption':
            self.weights = {
                'xgboost': 0.60,
                'gru': 0.30,
                'rf': 0.10
            }
        else:  # price
            self.weights = {
                'xgboost': 0.50,
                'gru': 0.35,
                'rf': 0.15
            }
        
        # Models dictionary
        self.models = {}
        self.scalers = {}
        self.gru_model = None
        
    def load_models(self):
        """
        Load all required models (XGB, GRU, RF) and their target/feature scalers from disk.
        
        Inputs: None, relies on serialized output paths (`outputs/models/*.pkl`).
        Outputs: Modifies `self.models`, `self.gru_model`, and `self.scalers` in-place.
        """
        print(f"\n📦 Loading ensemble components for {self.target_type}...")
        
        try:
            # Load XGBoost model
            model_name = f'xgb_{self.target_type}'
            self.models['xgboost'] = pickle.load(
                open(f'outputs/models/{model_name}.pkl', 'rb')
            )
            print(f"  ✓ XGBoost loaded (weight: {self.weights['xgboost']:.0%})")
            
            # Load Random Forest model
            model_name = f'rf_{self.target_type}'
            self.models['rf'] = pickle.load(
                open(f'outputs/models/{model_name}.pkl', 'rb')
            )
            print(f"  ✓ Random Forest loaded (weight: {self.weights['rf']:.0%})")
            
            # Load GRU model
            if self.target_type == 'consumption':
                checkpoint = torch.load(
                    'outputs/models/gru_consumption_final.pt',
                    map_location=self.device
                )
            else:
                checkpoint = torch.load(
                    'outputs/models/gru_price_final.pt',
                    map_location=self.device
                )
            
            input_size = 82  # From training
            self.gru_model = EnergyGRU(input_size=input_size, output_steps=24).to(self.device)
            self.gru_model.load_state_dict(checkpoint['model_state_dict'])
            self.gru_model.eval()
            print(f"  ✓ GRU loaded (weight: {self.weights['gru']:.0%})")
            
            # Load scalers
            self.scalers['scaler_X'] = pickle.load(open('outputs/models/scaler_X.pkl', 'rb'))
            
            if self.target_type == 'consumption':
                self.scalers['scaler_target'] = pickle.load(
                    open('outputs/models/scaler_cons.pkl', 'rb')
                )
            else:
                self.scalers['scaler_target'] = pickle.load(
                    open('outputs/models/scaler_price.pkl', 'rb')
                )
            
            self.scalers['imputer'] = pickle.load(open('outputs/models/imputer.pkl', 'rb'))
            self.scalers['svm_scaler_X'] = pickle.load(
                open('outputs/models/svm_scaler_X.pkl', 'rb')
            )
            
            print(f"  ✓ Scalers loaded\n✅ Ensemble ready for {self.target_type} prediction!")
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            raise
    
    def predict(self, X_raw, X_seq_scaled, return_components=False):
        """
        Generate ensemble predictions aggregating outputs from 3+ models combined into weighted vectors.
        
        Inputs:
          - X_raw: np.ndarray, shape [n_samples, n_features], raw unscaled tabular features.
          - X_seq_scaled: np.ndarray, shape [n_samples_seq, seq_length=48, n_features], sequence input for GRU.
          - return_components: bool, flag whether to return model components.
        
        Outputs:
          - ensemble_pred: np.ndarray, shape [n_samples, 24] or [n_samples_seq, 24].
          - components: dict, returned if return_components=True.
          
        ML Principle: Ensemble voting logic indicating how predictions from XGBoost, RF, GRU are 
        aggregated using weighted averaging, incorporating confidence thresholding proxy values.
        """
        
        # Preprocess X_raw (Handle missing values/imputation before model forward passes)
        X_clean = np.where(np.isinf(X_raw), np.nan, X_raw)
        X_clean = self.scalers['imputer'].transform(X_clean)
        
        predictions = {}
        
        # 1. XGBoost prediction
        try:
            predictions['xgboost'] = self.models['xgboost'].predict(X_clean)
        except Exception as e:
            print(f"⚠️ XGBoost prediction error: {e}")
            predictions['xgboost'] = np.zeros((X_clean.shape[0], 24))
        
        # 2. Random Forest prediction
        try:
            predictions['rf'] = self.models['rf'].predict(X_clean)
        except Exception as e:
            print(f"⚠️ Random Forest prediction error: {e}")
            predictions['rf'] = np.zeros((X_clean.shape[0], 24))
        
        # 3. GRU prediction (uses sequences)
        try:
            X_seq_tensor = torch.FloatTensor(X_seq_scaled).to(self.device)
            with torch.no_grad():
                preds_scaled = self.gru_model(X_seq_tensor).cpu().numpy()
            predictions['gru'] = self.scalers['scaler_target'].inverse_transform(preds_scaled)
        except Exception as e:
            print(f"⚠️ GRU prediction error: {e}")
            # GRU has fewer samples due to sequence, adjust XGB/RF
            predictions['gru'] = np.zeros((X_seq_scaled.shape[0], 24))
        
        # 4. Combine predictions
        # Align dimensions: GRU has fewer samples, so use GRU's length
        n_samples = predictions['gru'].shape[0]
        xgb_aligned = predictions['xgboost'][-n_samples:] if len(predictions['xgboost']) > n_samples else predictions['xgboost']
        rf_aligned = predictions['rf'][-n_samples:] if len(predictions['rf']) > n_samples else predictions['rf']
        
        # Ensemble voting logic (how predictions from XGB, RF, and GRU are aggregated)
        # Weighting acts as a confidence thresholding strategy based on previous validation performance
        ensemble_pred = (
            self.weights['xgboost'] * xgb_aligned +
            self.weights['gru'] * predictions['gru'] +
            self.weights['rf'] * rf_aligned
        )
        
        if return_components:
            return ensemble_pred, {
                'xgboost': xgb_aligned,
                'gru': predictions['gru'],
                'rf': rf_aligned
            }
        
        return ensemble_pred


def create_ensemble_consumption():
    """Factory function for consumption ensemble"""
    ensemble = EnsembleModel(target_type='consumption')
    ensemble.load_models()
    return ensemble


def create_ensemble_price():
    """Factory function for price ensemble"""
    ensemble = EnsembleModel(target_type='price')
    ensemble.load_models()
    return ensemble


if __name__ == '__main__':
    print("=" * 80)
    print("ENSEMBLE MODEL INITIALIZATION TEST")
    print("=" * 80)
    
    # Test consumption ensemble
    print("\nTesting Consumption Ensemble:")
    ens_cons = create_ensemble_consumption()
    print(f"✓ Consumption ensemble initialized")
    
    # Test price ensemble
    print("\nTesting Price Ensemble:")
    ens_price = create_ensemble_price()
    print(f"✓ Price ensemble initialized")
    
    print("\n" + "=" * 80)
    print("✅ ENSEMBLE MODELS READY!")
    print("=" * 80)
