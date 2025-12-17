# test_nan_fix.py
import numpy as np
import sys
sys.path.insert(0, 'src')

from c50 import C50Stump

# Dataset avec NaN
X_train = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
y_train = np.array([0, 0, 0, 1, 1, 1])

X_test = np.array([[2.5], [np.nan], [np.nan], [5.5]])
y_test = np.array([0, 0, 1, 1])

# Entraîner
stump = C50Stump(handle_missing=True)
stump.fit(X_train, y_train)

print(f"Missing strategy: {stump.missing_strategy_}")

# Prédire plusieurs fois
print("\nTest de stabilité (devrait être identique):")
for i in range(5):
    y_pred = stump.predict(X_test)
    print(f"  Run {i+1}: {y_pred}")

# Accuracy
acc = stump.score(X_test, y_test)
print(f"\nAccuracy: {acc:.2%}")