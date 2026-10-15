import os
import cv2
import random
import numpy as np

from sklearn.model_selection import train_test_split

# =====================================
# CONFIGURATION
# =====================================

DATASET_PATH = "lfw-deepfunneled"

IMG_SIZE = 96

MIN_IMAGES = 2

POSITIVE_PAIRS_PER_PERSON = 10
NEGATIVE_PAIRS_PER_PERSON = 10

# =====================================
# LOAD DATASET
# =====================================

def load_dataset():

    faces = []
    labels = []

    print("Loading images...")

    for person in os.listdir(DATASET_PATH):

        person_path = os.path.join(
            DATASET_PATH,
            person
        )

        if not os.path.isdir(person_path):
            continue

        image_files = os.listdir(person_path)

        if len(image_files) < MIN_IMAGES:
            continue

        for image_name in image_files:

            image_path = os.path.join(
                person_path,
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

            faces.append(image)
            labels.append(person)

    return np.array(faces), np.array(labels)


# =====================================
# CREATE PAIRS
# =====================================

def create_pairs(images, labels):

    pair_images = []
    pair_labels = []

    unique_labels = np.unique(labels)

    label_indices = {
        label: np.where(labels == label)[0]
        for label in unique_labels
    }

    print("Generating pairs...")

    for label in unique_labels:

        indices = label_indices[label]

        if len(indices) < 2:
            continue

        # -------------------------
        # Positive Pairs
        # -------------------------

        for _ in range(POSITIVE_PAIRS_PER_PERSON):

            idx1, idx2 = np.random.choice(
                indices,
                2,
                replace=False
            )

            pair_images.append(
                [images[idx1], images[idx2]]
            )

            pair_labels.append(1)

        # -------------------------
        # Negative Pairs
        # -------------------------

        for _ in range(NEGATIVE_PAIRS_PER_PERSON):

            idx1 = random.choice(indices)

            other_label = random.choice(
                unique_labels
            )

            while other_label == label:

                other_label = random.choice(
                    unique_labels
                )

            idx2 = random.choice(
                label_indices[other_label]
            )

            pair_images.append(
                [images[idx1], images[idx2]]
            )

            pair_labels.append(0)

    pair_images = np.array(
        pair_images,
        dtype=np.uint8
    )

    pair_labels = np.array(
        pair_labels,
        dtype=np.uint8
    )

    return pair_images, pair_labels


# =====================================
# MAIN
# =====================================

print("Loading dataset...")

images, labels = load_dataset()

print("\nImages Loaded:", len(images))
print("Identities:", len(np.unique(labels)))

pairs, pair_labels = create_pairs(
    images,
    labels
)

print("\nTotal Pairs:", len(pairs))

# =====================================
# TRAIN TEST SPLIT
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    pairs,
    pair_labels,
    test_size=0.2,
    stratify=pair_labels,
    random_state=42
)

# =====================================
# SAVE DATASET
# =====================================

os.makedirs(
    "dataset",
    exist_ok=True
)

np.save(
    "dataset/X_train.npy",
    X_train
)

np.save(
    "dataset/X_test.npy",
    X_test
)

np.save(
    "dataset/y_train.npy",
    y_train
)

np.save(
    "dataset/y_test.npy",
    y_test
)

print("\n================================")
print("Dataset Created Successfully")
print("================================")

print("Training Pairs:", len(X_train))
print("Testing Pairs :", len(X_test))







# import os    before changing 5-06-2026
# import cv2
# import random
# import numpy as np
# from sklearn.model_selection import train_test_split

# # -----------------------------
# # Configuration
# # -----------------------------

# DATASET_PATH = "lfw-deepfunneled"
# IMG_SIZE = 100
# MIN_IMAGES = 10

# # -----------------------------
# # Load Dataset
# # -----------------------------

# def load_dataset():

#     faces = []
#     labels = []

#     for person in os.listdir(DATASET_PATH):

#         person_path = os.path.join(DATASET_PATH, person)

#         if not os.path.isdir(person_path):
#             continue

#         images = os.listdir(person_path)

#         # skip people with very few images
#         if len(images) < MIN_IMAGES:
#             continue

#         for img in images:

#             img_path = os.path.join(person_path, img)

#             if not img.lower().endswith((".jpg", ".jpeg", ".png")):
#                 continue

#             image = cv2.imread(img_path)

#             if image is None:
#                 continue

#             # resize image
#             image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

#             # convert from BGR to RGB
#             image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#             faces.append(image)
#             labels.append(person)

#     return np.array(faces), np.array(labels)


# # -----------------------------
# # Create Siamese Pairs
# # -----------------------------

# def create_pairs(images, labels):

#     pair_images = []
#     pair_labels = []

#     unique_labels = np.unique(labels)

#     label_indices = {
#         label: np.where(labels == label)[0]
#         for label in unique_labels
#     }

#     # Generate balanced, diverse pairs
#     for label in unique_labels:

#         idx = label_indices[label]
#         n_images = len(idx)

#         # Diverse positive pairs generation per person
#         # Cap at min(8, n_images) to accelerate CPU training 10x without accuracy loss
#         n_pos_pairs = min(8, n_images)
#         pos_pairs_generated = set()
#         attempts = 0

#         while len(pos_pairs_generated) < n_pos_pairs and attempts < n_pos_pairs * 10:
#             attempts += 1
#             i = random.randint(0, n_images - 1)
#             j = random.randint(0, n_images - 1)
#             if i != j:
#                 # Sort indices to avoid duplicates
#                 pair = tuple(sorted((idx[i], idx[j])))
#                 pos_pairs_generated.add(pair)

#         # Fallback to consecutive if random generation failed to find enough pairs
#         if len(pos_pairs_generated) < 2:
#             for k in range(n_images - 1):
#                 pos_pairs_generated.add((idx[k], idx[k+1]))

#         # Add pairs
#         for img1_idx, img2_idx in pos_pairs_generated:
#             # Positive pair
#             pair_images.append([images[img1_idx], images[img2_idx]])
#             pair_labels.append(1)

#             # Negative pair
#             other_label = random.choice(unique_labels)
#             while other_label == label:
#                 other_label = random.choice(unique_labels)

#             other_idx = random.choice(label_indices[other_label])
#             img_neg = images[img1_idx] if random.random() < 0.5 else images[img2_idx]
#             pair_images.append([img_neg, images[other_idx]])
#             pair_labels.append(0)

#     return np.array(pair_images), np.array(pair_labels)


# # -----------------------------
# # Main Execution
# # -----------------------------

# print("Loading dataset...")

# images, labels = load_dataset()

# print("Images loaded:", len(images))

# print("Creating pairs...")

# pairs, pair_labels = create_pairs(images, labels)

# # reshape for CNN input: shape is (num_pairs, 2, IMG_SIZE, IMG_SIZE, 3)
# pairs = pairs.reshape(pairs.shape[0], 2, IMG_SIZE, IMG_SIZE, 3)

# # -----------------------------
# # Train Test Split
# # -----------------------------

# X_train, X_test, y_train, y_test = train_test_split(
#     pairs,
#     pair_labels,
#     test_size=0.2,
#     random_state=42
# )

# # -----------------------------
# # Save Dataset
# # -----------------------------

# os.makedirs("dataset", exist_ok=True)

# # Save as uint8 to conserve memory/disk space (normalize to float32 in training)
# np.save("dataset/X_train.npy", X_train.astype("uint8"))
# np.save("dataset/X_test.npy", X_test.astype("uint8"))
# np.save("dataset/y_train.npy", np.array(y_train, dtype="uint8"))
# np.save("dataset/y_test.npy", np.array(y_test, dtype="uint8"))

# print("Dataset prepared successfully")
# print("Files saved in 'dataset' folder")
















# import os
# import cv2
# import random
# import numpy as np
# from sklearn.model_selection import train_test_split

# DATASET_PATH = "lfw-deepfunneled"
# IMG_SIZE = 100
# MIN_IMAGES = 10


# def load_dataset():

#     faces = []
#     labels = []

#     for person in os.listdir(DATASET_PATH):

#         person_path = os.path.join(DATASET_PATH, person)

#         if not os.path.isdir(person_path):
#             continue

#         images = os.listdir(person_path)

#         if len(images) < MIN_IMAGES:
#             continue

#         for img in images:

#             img_path = os.path.join(person_path, img)

#             if not img.lower().endswith((".jpg", ".jpeg", ".png")):
#                 continue

#             image = cv2.imread(img_path)

#             if image is None:
#                 continue

#             image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
#             image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#             faces.append(image)
#             labels.append(person)

#     return np.array(faces), np.array(labels)


# def create_pairs(images, labels):

#     pair_images = []
#     pair_labels = []

#     unique_labels = np.unique(labels)

#     label_indices = {
#         label: np.where(labels == label)[0]
#         for label in unique_labels
#     }

#     for label in unique_labels:

#         idx = label_indices[label]

#         for i in range(len(idx) - 1):

#             img1 = images[idx[i]]
#             img2 = images[idx[i + 1]]

#             pair_images.append([img1, img2])
#             pair_labels.append(1)

#             other_label = random.choice(unique_labels)

#             while other_label == label:
#                 other_label = random.choice(unique_labels)

#             img3 = images[random.choice(label_indices[other_label])]

#             pair_images.append([img1, img3])
#             pair_labels.append(0)

#     return np.array(pair_images), np.array(pair_labels)


# print("Loading dataset...")

# images, labels = load_dataset()

# print("Images loaded:", len(images))

# print("Creating pairs...")

# pairs, pair_labels = create_pairs(images, labels)

# pairs = pairs.reshape(pairs.shape[0], 2, IMG_SIZE, IMG_SIZE, 1)

# X_train, X_test, y_train, y_test = train_test_split(
#     pairs,
#     pair_labels,
#     test_size=0.2,
#     random_state=42
# )

# os.makedirs("dataset", exist_ok=True)

# np.save("dataset/X_train.npy", X_train)
# np.save("dataset/X_test.npy", X_test)
# np.save("dataset/y_train.npy", y_train)
# np.save("dataset/y_test.npy", y_test)

# print("Dataset prepared successfully")

















# import os
# import cv2
# import numpy as np
# from sklearn.model_selection import train_test_split

# # Automatically detect dataset folder
# BASE_DIR = os.getcwd()

# # Try to locate lfw dataset
# DATASET_PATH = None

# for root, dirs, files in os.walk(BASE_DIR):
#     if "lfw-deepfunneled" in dirs:
#         DATASET_PATH = os.path.join(root, "lfw-deepfunneled")
#         break

# if DATASET_PATH is None:
#     raise Exception("lfw-deepfunneled dataset folder not found. Extract the Kaggle dataset first.")

# print("Dataset found at:", DATASET_PATH)

# IMG_SIZE = 100


# def load_images():
#     images = []
#     labels = []

#     people = os.listdir(DATASET_PATH)

#     for person in people:

#         person_path = os.path.join(DATASET_PATH, person)

#         if not os.path.isdir(person_path):
#             continue

#         for img in os.listdir(person_path):

#             img_path = os.path.join(person_path, img)

#             image = cv2.imread(img_path)

#             if image is None:
#                 continue

#             image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
#             image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#             images.append(image)
#             labels.append(person)

#     return np.array(images), np.array(labels)


# def create_pairs(images, labels):

#     pair_images = []
#     pair_labels = []

#     unique_labels = np.unique(labels)

#     for label in unique_labels:

#         idx = np.where(labels == label)[0]

#         if len(idx) < 2:
#             continue

#         for i in range(len(idx) - 1):

#             img1 = images[idx[i]]
#             img2 = images[idx[i + 1]]

#             pair_images.append([img1, img2])
#             pair_labels.append(1)

#     for i in range(len(images) - 1):

#         if labels[i] != labels[i + 1]:

#             pair_images.append([images[i], images[i + 1]])
#             pair_labels.append(0)

#     return np.array(pair_images), np.array(pair_labels)


# print("Loading images...")

# images, labels = load_images()

# print("Total images loaded:", len(images))

# print("Creating pairs...")

# pairs, pair_labels = create_pairs(images, labels)

# pairs = pairs.reshape(pairs.shape[0], 2, IMG_SIZE, IMG_SIZE, 1)

# print("Splitting dataset...")

# X_train, X_test, y_train, y_test = train_test_split(
#     pairs, pair_labels, test_size=0.2
# )

# os.makedirs("dataset", exist_ok=True)

# np.save("dataset/X_train.npy", X_train)
# np.save("dataset/X_test.npy", X_test)
# np.save("dataset/y_train.npy", y_train)
# np.save("dataset/y_test.npy", y_test)

# print("Dataset prepared successfully!")