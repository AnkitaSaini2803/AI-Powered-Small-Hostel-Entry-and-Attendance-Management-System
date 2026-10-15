import numpy as np
import tensorflow as tf

from tensorflow.keras.models import load_model

def l1_distance(tensors):
    x,y=tensors
    return tf.abs(x-y)

model = load_model(
    "face_recognition_model.keras",
    custom_objects={
        "l1_distance": l1_distance
    },
    compile=False
)

X_test = np.load("dataset/X_test.npy").astype("float32")/255.0
y_test = np.load("dataset/y_test.npy")

predictions = model.predict(
    [X_test[:,0], X_test[:,1]],
    batch_size=64
).flatten()

positive_scores = predictions[y_test == 1]
negative_scores = predictions[y_test == 0]

print("\nPositive Pairs")
print("Mean:", np.mean(positive_scores))
print("Min :", np.min(positive_scores))
print("Max :", np.max(positive_scores))

print("\nNegative Pairs")
print("Mean:", np.mean(negative_scores))
print("Min :", np.min(negative_scores))
print("Max :", np.max(negative_scores))