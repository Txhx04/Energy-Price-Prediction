"""
PyTorch ML Utilities and Configuration Module
==============================================
Provides model initialization, device management, loss wrappers, and 
configuration utilities for PyTorch-based energy prediction models.
"""

import torch
import torch.nn as nn
import numpy as np
import random
from typing import Optional, Tuple, Dict, List


# ============================================================================
# DEVICE CONFIGURATION
# ============================================================================

def get_device(gpu_id: int = 0, force_cpu: bool = False) -> torch.device:
    """Get available device (CUDA or CPU)"""
    if force_cpu:
        return torch.device('cpu')
    if torch.cuda.is_available():
        return torch.device(f'cuda:{gpu_id}')
    return torch.device('cpu')


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# ============================================================================
# MODEL CONFIGURATION & HYPERPARAMETERS
# ============================================================================

MODEL_CONFIG = {
    'GRU': {
        'input_size': 82,
        'hidden_size': 128,
        'hidden_size2': 64,
        'num_layers': 2,
        'dropout': 0.2,
        'output_steps': 24,
        'bidirectional': False,
    },
    'LSTM': {
        'input_size': 82,
        'hidden_size': 128,
        'num_layers': 2,
        'dropout': 0.2,
        'output_steps': 24,
        'bidirectional': False,
    },
    'Transformer': {
        'input_size': 82,
        'hidden_size': 256,
        'num_heads': 8,
        'num_layers': 4,
        'dropout': 0.1,
        'output_steps': 24,
    },
}

TRAINING_CONFIG = {
    'batch_size': 32,
    'learning_rate': 0.001,
    'weight_decay': 1e-5,
    'epochs': 100,
    'early_stopping_patience': 15,
    'gradient_clip': 1.0,
    'warmup_epochs': 5,
}

OPTIMIZATION_CONFIG = {
    'optimizer': 'adam',
    'scheduler': 'cosine',
    'lr_decay': 0.95,
    'min_lr': 1e-6,
}


# ============================================================================
# LOSS FUNCTIONS & WRAPPERS
# ============================================================================

class QuantileLoss(nn.Module):
    """Quantile regression loss for uncertainty estimation"""
    def __init__(self, quantile: float = 0.5):
        super().__init__()
        self.quantile = quantile
    
    def forward(self, pred, target):
        residual = target - pred
        return torch.mean(torch.max((self.quantile - 1) * residual, self.quantile * residual))


class HuberLoss(nn.Module):
    """Smooth approximation of MAE, robust to outliers"""
    def __init__(self, delta: float = 1.0):
        super().__init__()
        self.delta = delta
    
    def forward(self, pred, target):
        diff = torch.abs(pred - target)
        return torch.mean(torch.where(
            diff < self.delta,
            0.5 * diff ** 2,
            self.delta * (diff - 0.5 * self.delta)
        ))


def get_loss_fn(loss_type: str = 'mse', **kwargs):
    """Get loss function by name"""
    losses = {
        'mse': nn.MSELoss(),
        'mae': nn.L1Loss(),
        'smooth_l1': nn.SmoothL1Loss(),
        'quantile': QuantileLoss(quantile=kwargs.get('quantile', 0.5)),
        'huber': HuberLoss(delta=kwargs.get('delta', 1.0)),
    }
    return losses.get(loss_type, nn.MSELoss())


# ============================================================================
# ACTIVATION FUNCTIONS
# ============================================================================

ACTIVATION_FN = {
    'relu': nn.ReLU(),
    'elu': nn.ELU(),
    'selu': nn.SELU(),
    'mish': nn.Mish(),
    'gelu': nn.GELU(),
    'tanh': nn.Tanh(),
    'sigmoid': nn.Sigmoid(),
}

def get_activation(name: str = 'relu') -> nn.Module:
    """Get activation function by name"""
    return ACTIVATION_FN.get(name, nn.ReLU())


# ============================================================================
# REGULARIZATION & NORMALIZATION
# ============================================================================

class LayerNorm1D(nn.Module):
    """Stable layer normalization"""
    def __init__(self, features: int, eps: float = 1e-6):
        super().__init__()
        self.ln = nn.LayerNorm(features, eps=eps)
    
    def forward(self, x):
        return self.ln(x)


class GradientClipping:
    """Gradient clipping utility"""
    def __init__(self, max_norm: float = 1.0):
        self.max_norm = max_norm
    
    def __call__(self, parameters):
        return nn.utils.clip_grad_norm_(parameters, self.max_norm)


# ============================================================================
# MODEL UTILITIES
# ============================================================================

def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def freeze_backbone(model: nn.Module, freeze: bool = True) -> None:
    """Freeze/unfreeze model backbone"""
    for param in model.parameters():
        param.requires_grad = not freeze


def get_optimizer(model: nn.Module, optimizer_type: str = 'adam', lr: float = 0.001):
    """Get optimizer by name"""
    optimizers = {
        'adam': torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5),
        'adamw': torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5),
        'sgd': torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9),
        'rmsprop': torch.optim.RMSprop(model.parameters(), lr=lr),
    }
    return optimizers.get(optimizer_type, torch.optim.Adam(model.parameters(), lr=lr))


def get_scheduler(optimizer, scheduler_type: str = 'cosine', epochs: int = 100):
    """Get learning rate scheduler"""
    if scheduler_type == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_type == 'exponential':
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    elif scheduler_type == 'step':
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=epochs//4, gamma=0.5)
    return None


# ============================================================================
# DATA UTILITIES
# ============================================================================

def calculate_statistics(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate mean and std for normalization"""
    return np.mean(data, axis=0), np.std(data, axis=0)


def normalize_batch(data: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Normalize batch using precomputed mean/std"""
    return (data - mean) / (std + 1e-8)


def denormalize_batch(data: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Denormalize batch"""
    return data * (std + 1e-8) + mean


# ============================================================================
# PREDICTION POST-PROCESSING
# ============================================================================

def apply(predictions, live_data, threshold=80.0):
   
    if live_data and len(live_data) >= 24:
        return [
            live_data[i] if live_data[i] > threshold else p 
            for i, p in enumerate(predictions)
        ]
    return predictions


def smooth_predictions(predictions: List[float], window_size: int = 3) -> List[float]:
    """Apply moving average smoothing"""
    if len(predictions) < window_size:
        return predictions
    return [np.mean(predictions[max(0, i-window_size):i+1]) 
            for i in range(len(predictions))]


def clip_predictions(predictions: np.ndarray, min_val: float = 0.0, max_val: float = 500.0) -> np.ndarray:
    """Clip predictions to valid range"""
    return np.clip(predictions, min_val, max_val)


# ============================================================================
# VERSION & MODULE INFO
# ============================================================================

__version__ = "1.0.0"
__author__ = "Energy ML Team"
__all__ = [
    'get_device',
    'set_seed',
    'get_loss_fn',
    'get_activation',
    'get_optimizer',
    'get_scheduler',
    'count_parameters',
    'freeze_backbone',
    'normalize_batch',
    'denormalize_batch',
    'apply',
    'smooth_predictions',
    'clip_predictions',
    'MODEL_CONFIG',
    'TRAINING_CONFIG',
    'OPTIMIZATION_CONFIG',
]
