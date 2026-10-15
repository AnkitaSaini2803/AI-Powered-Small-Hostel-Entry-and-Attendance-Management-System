import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load test data
X_test = np.load("dataset/X_test.npy")
y_test = np.load("dataset/y_test.npy")

# Load trained model
model = load_model("face_recognition_model.keras", compile=False)

# Get distances for all test pairs
distances = model.predict(
    [X_test[:, 0], X_test[:, 1]],
    verbose=1
)

# Try every threshold and print accuracy
print("\nThreshold | Accuracy")
print("-" * 25)

best_acc = 0
best_threshold = 0.5

for t in np.arange(0.2, 1.5, 0.05):
    preds = (distances < t).astype(int).flatten()
    acc   = (preds == y_test).mean()
    marker = " ← best" if acc > best_acc else ""
    print(f"  {t:.2f}    |  {acc:.4f}{marker}")
    if acc > best_acc:
        best_acc = acc
        best_threshold = t

print(f"\nUse THRESHOLD = {best_threshold:.2f} in real_time_face_recognition.py")