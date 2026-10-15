import os
import tensorflow as tf

MODEL_PATH = "face_siamese.keras"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "Model not found! Put face_siamese.keras in the SAME folder as new.py"
    )

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("Model loaded ✔")