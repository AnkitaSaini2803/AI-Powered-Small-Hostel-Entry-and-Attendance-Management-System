import os
import cv2
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

IMG_SIZE = 96

DATABASE_PATH = "database"

embedding_model = load_model(
    "embedding_model.keras",
    compile=False
)

embeddings = []
labels = []

for person in os.listdir(DATABASE_PATH):

    person_folder = os.path.join(
        DATABASE_PATH,
        person
    )

    if not os.path.isdir(person_folder):
        continue

    print("Processing:", person)

    for image_name in os.listdir(person_folder):

        image_path = os.path.join(
            person_folder,
            image_name
        )

        image = cv2.imread(image_path)

        if image is None:
            continue

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        image = cv2.resize(
            image,
            (IMG_SIZE, IMG_SIZE)
        )

        image = image.astype("float32")

        image = preprocess_input(image)

        embedding = embedding_model.predict(
            np.expand_dims(image, axis=0),
            verbose=0
        )[0]
        # Normalize embedding
        embedding = embedding / np.linalg.norm(embedding)  #added this 
        
        embeddings.append(embedding)
        labels.append(person)

embeddings = np.array(embeddings)
labels = np.array(labels)

np.save(
    "embeddings.npy",
    embeddings
)

np.save(
    "labels.npy",
    labels
)

print("\nEmbeddings Saved")
print("Total:", len(embeddings))