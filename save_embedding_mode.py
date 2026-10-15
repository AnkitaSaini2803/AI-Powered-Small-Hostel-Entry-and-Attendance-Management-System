import tensorflow as tf
from tensorflow.keras.models import load_model, Model

# ==================================
# L1 Distance
# ==================================

def l1_distance(vectors):
    x, y = vectors
    return tf.abs(x - y)

# ==================================
# Load Trained Siamese Model
# ==================================

model = load_model(
    "face_recognition_model.keras",
    custom_objects={
        "l1_distance": l1_distance
    },
    compile=False
)

# ==================================
# Get Embedding Network
# ==================================

embedding_model = model.get_layer(
    "EmbeddingNetwork"
)

embedding_model.save(
    "embedding_model.keras"
)

print("Embedding Model Saved")