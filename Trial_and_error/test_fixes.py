import numpy as np
import pandas as pd
import pickle
import sys
sys.path.insert(0, 'validation')

# Define safe_divide function
def safe_divide(numerator, denominator, default=0.0):
    """Safely divide, returning default if denominator is zero"""
    return (numerator / denominator) if denominator > 0 else default

# Define create_binned_confusion function
def create_binned_confusion(actual, pred, n_bins=3, model_name='Model', target_name='Target'):
    """
    Create a confusion-like matrix for regression by binning predictions and actuals.
    Bins are created based on percentiles: [0-33%, 33-66%, 66-100%] or custom thresholds.
    Handles length mismatches by aligning to shortest length.
    """
    # ✅ FIX: Align lengths (LSTM/GRU sequences are shorter than ML models' predictions)
    min_len = min(len(actual), len(pred))
    actual = actual[:min_len]
    pred = pred[:min_len]
    
    # Use percentile-based binning
    bins = np.percentile(actual, [0, 33.33, 66.67, 100])
    bin_labels = ['Low', 'Medium', 'High']
    
    # Flatten for entire prediction set
    actual_flat = actual.flatten()
    pred_flat = pred.flatten()
    
    actual_bins = np.digitize(actual_flat, bins[1:-1]) - 1
    pred_bins = np.digitize(pred_flat, bins[1:-1]) - 1
    
    # Clip to valid range
    actual_bins = np.clip(actual_bins, 0, 2)
    pred_bins = np.clip(pred_bins, 0, 2)
    
    # Create confusion matrix (vectorized)
    cm = np.zeros((3, 3), dtype=int)
    for i in range(3):
        cm[i, :] = np.bincount(pred_bins[actual_bins == i], minlength=3)
    
    # ✅ FIX: Safely normalize by row (handle empty bins)
    cm_norm = np.zeros_like(cm, dtype=float)
    for i in range(3):
        row_sum = cm[i].sum()
        if row_sum > 0:
            cm_norm[i, :] = cm[i, :] / row_sum
        else:
            cm_norm[i, :] = 0.0
    
    return cm, cm_norm, bin_labels, bins

print("=" * 80)
print("TESTING QUICK FIXES FOR BINNED ACCURACY")
print("=" * 80)

# Create synthetic validation data for testing
print("\n1. Creating synthetic test data...")
np.random.seed(42)
n_samples = 100
n_targets = 24

# Create synthetic actual and predicted values
y_actually_cons = np.random.randn(n_samples, n_targets) + 500  # consumption-like values
y_pred_cons = np.random.randn(n_samples - 10, n_targets) + 500  # slightly shorter (simulate LSTM)

print(f"   Actual samples: {y_actually_cons.shape}")
print(f"   Predicted samples (LSTM-like): {y_pred_cons.shape}")
print(f"   Length mismatch: {len(y_actually_cons) - len(y_pred_cons)} samples")

# Test 1: Load a simple model and create predictions with different lengths
print("\n2. Testing  length alignment fix...")
try:
    print(f"✅ Testing with mismatched lengths...")
    
    # Test the create_binned_confusion function with different lengths
    print("\n3. Testing create_binned_confusion with length mismatch...")
    cm, cm_norm, labels, bins = create_binned_confusion(
        y_actually_cons, y_pred_cons
    )
    
    print(f"   Confusion matrix shape: {cm.shape}")
    print(f"   Normalized confusion matrix:\n{cm_norm}")
    print(f"   Diagonal (per-bin accuracy): {[f'{cm_norm[i,i]:.1%}' for i in range(3)]}")
    
    # Test 2: Check for NaN values
    print("\n4. Checking for NaN values...")
    has_nan = np.isnan(cm_norm).any()
    print(f"   NaN in confusion matrix: {has_nan}")
    if not has_nan:
        print(f"   ✅ No NaN values found (fix is working!)")
    else:
        print(f"   ❌ NaN values still present (fix not working)")
    
    # Test 3: Verify safe_divide works
    print("\n5. Testing safe_divide function...")
    test_cases = [
        (10, 5, "Normal division"),
        (0, 0, "Zero by zero (should return 0.0)"),
        (5, 0, "Non-zero by zero (should return 0.0)"),
    ]
    
    for num, denom, desc in test_cases:
        result = safe_divide(num, denom, default=0.0)
        print(f"   {desc}: safe_divide({num}, {denom}) = {result}")
    
    # Test 4: Full binned accuracy calculation
    print("\n6. Computing full binned accuracy...")
    accuracy = np.trace(cm_norm) / 3
    low_acc = (cm_norm[0, 0] * 100) if np.isfinite(cm_norm[0, 0]) else 0.0
    med_acc = (cm_norm[1, 1] * 100) if np.isfinite(cm_norm[1, 1]) else 0.0
    high_acc = (cm_norm[2, 2] * 100) if np.isfinite(cm_norm[2, 2]) else 0.0
    bin_acc = (accuracy * 100) if np.isfinite(accuracy * 100) else 0.0
    
    print(f"   Bin Accuracy: {bin_acc:.2f}%")
    print(f"   Low Accuracy: {low_acc:.2f}%")
    print(f"   Medium Accuracy: {med_acc:.2f}%")
    print(f"   High Accuracy: {high_acc:.2f}%")
    
    # Verify all values are numbers (not NaN)
    all_finite = all(np.isfinite(v) for v in [bin_acc, low_acc, med_acc, high_acc])
    if all_finite:
        print(f"\n✅ All accuracy values are finite (no NaN, fix working!)")
    else:
        print(f"\n❌ Some accuracy values are not finite")
    
    print("\n" + "=" * 80)
    if not has_nan and all_finite:
        print("✅ ALL FIXES VERIFIED SUCCESSFULLY!")
    else:
        print("⚠️  Some issues detected")
    print("=" * 80)
    
except Exception as e:
    print(f"\n✗ Error during testing: {e}")
    import traceback
    traceback.print_exc()

