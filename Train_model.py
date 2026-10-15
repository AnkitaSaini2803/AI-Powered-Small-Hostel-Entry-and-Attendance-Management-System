import os
import numpy as np

# ==========================================
# MEMORY OPTIMIZATION
# ==========================================

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf

tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

from tensorflow.keras.layers import (
    Input,
    Dense,
    Lambda,
    Dropout
)

from tensorflow.keras.models import Model

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ==========================================
# LOAD DATASET
# ==========================================

print("Loading Dataset...")

X_train = np.load(
    "dataset/X_train.npy",
    mmap_mode="r"
)

X_test = np.load(
    "dataset/X_test.npy",
    mmap_mode="r"
)

y_train = np.load(
    "dataset/y_train.npy"
)

y_test = np.load(
    "dataset/y_test.npy"
)

print("Train Shape :", X_train.shape)
print("Test Shape  :", X_test.shape)

IMG_SIZE = X_train.shape[2]

# ==========================================
# PREPROCESSING
# ==========================================

def preprocess_images(images):

    images = images.astype("float32")

    images = preprocess_input(images)

    return images

# ==========================================
# L1 DISTANCE
# ==========================================

def l1_distance(vectors):

    x, y = vectors

    return tf.abs(x - y)

# ==========================================
# EMBEDDING NETWORK
# ==========================================

def build_embedding_network():

    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        pooling="avg"
    )

    # Freeze MobileNet completely
    base_model.trainable = False

    input_layer = Input(
        shape=(IMG_SIZE, IMG_SIZE, 3)
    )

    x = base_model(input_layer)

    x = Dense(
        128,
        activation="relu"
    )(x)

    x = Dropout(
        0.3
    )(x)

    x = Dense(
        64,
        activation="relu"
    )(x)

    return Model(
        input_layer,
        x,
        name="EmbeddingNetwork"
    )

# ==========================================
# BUILD SIAMESE NETWORK
# ==========================================

embedding_network = build_embedding_network()

input_a = Input(
    shape=(IMG_SIZE, IMG_SIZE, 3)
)

input_b = Input(
    shape=(IMG_SIZE, IMG_SIZE, 3)
)

embedding_a = embedding_network(input_a)

embedding_b = embedding_network(input_b)

distance = Lambda(
    l1_distance,
    name="L1_Distance"
)(
    [embedding_a, embedding_b]
)

output = Dense(
    1,
    activation="sigmoid"
)(
    distance
)

model = Model(
    inputs=[input_a, input_b],
    outputs=output,
    name="SiameseNetwork"
)

# ==========================================
# COMPILE
# ==========================================

model.compile(
    optimizer=Adam(
        learning_rate=1e-4
    ),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ==========================================
# PREPROCESS DATA
# ==========================================

print("\nPreprocessing Images...\n")

X1_train = preprocess_images(
    np.array(X_train[:, 0])
)

X2_train = preprocess_images(
    np.array(X_train[:, 1])
)

X1_test = preprocess_images(
    np.array(X_test[:, 0])
)

X2_test = preprocess_images(
    np.array(X_test[:, 1])
)

# ==========================================
# CALLBACKS
# ==========================================

early_stop = EarlyStopping(
    monitor="val_accuracy",
    patience=3, #2
    restore_best_weights=True,
    verbose=1
)

checkpoint = ModelCheckpoint(
    "best_weights.weights.h5",
    monitor="val_accuracy",
    save_best_only=True,
    save_weights_only=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=1,
    min_lr=1e-6,
    verbose=1
)

# ==========================================
# TRAIN
# ==========================================

print("\nTraining Started...\n")

history = model.fit(
    [X1_train, X2_train],
    y_train,
    validation_data=(
        [X1_test, X2_test],
        y_test
    ),
    batch_size=2,
    epochs=8, #2,
    callbacks=[
        early_stop,
        checkpoint,
        reduce_lr
    ]
)

# ==========================================
# LOAD BEST WEIGHTS
# ==========================================

print("\nLoading Best Weights...\n")

model.load_weights(
    "best_weights.weights.h5"
)

# ==========================================
# EVALUATE
# ==========================================

print("\nEvaluating Model...\n")

loss, accuracy = model.evaluate(
    [X1_test, X2_test],
    y_test,
    verbose=1
)

print("\n======================")
print("Test Loss     :", loss)
print("Test Accuracy :", accuracy)
print("======================")

# ==========================================
# FIND BEST THRESHOLD
# ==========================================

print("\nFinding Best Threshold...\n")

predictions = model.predict(
    [X1_test, X2_test],
    verbose=1
).flatten()

best_acc = 0
best_threshold = 0

for threshold in np.arange(
    0.10,
    0.91,
    0.01
):

    pred_labels = (
        predictions >= threshold
    ).astype(int)

    acc = np.mean(
        pred_labels == y_test
    )

    if acc > best_acc:

        best_acc = acc
        best_threshold = threshold

print("\n======================")
print("BEST THRESHOLD :", best_threshold)
print("BEST ACCURACY  :", best_acc)
print("======================")

with open(
    "best_threshold.txt",
    "w"
) as f:

    f.write(
        str(best_threshold)
    )

# ==========================================
# SAVE FINAL MODEL
# ==========================================

model.save(
    "face_recognition_model.keras"
)

print("\nModel Saved Successfully!")
print("Threshold Saved Successfully!")







# import os   79 % Accu but error  in other file 
# import cv2
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras import layers, Model
# from sklearn.model_selection import train_test_split
# import random

# # =========================
# # CONFIG
# # =========================
# IMG_SIZE = 96
# DATASET_PATH = "/content/lfw_data/lfw-deepfunneled"

# print("Loading dataset...")

# # =========================
# # LOAD DATASET (FAST + SAFE)
# # =========================
# def load_images(dataset_path):
#     data = {}

#     for person in os.listdir(dataset_path):
#         person_path = os.path.join(dataset_path, person)
#         if not os.path.isdir(person_path):
#             continue

#         imgs = []
#         for img_name in os.listdir(person_path):
#             img_path = os.path.join(person_path, img_name)

#             img = cv2.imread(img_path)
#             if img is None:
#                 continue

#             img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
#             img = img.astype("float32") / 255.0
#             imgs.append(img)

#         if len(imgs) >= 2:
#             data[person] = imgs

#     return data


# data = load_images(DATASET_PATH)
# print("Classes:", len(data))

# people = list(data.keys())

# # =========================
# # TRAIN / TEST SPLIT (BY IDENTITY)
# # =========================
# train_people, test_people = train_test_split(people, test_size=0.2, random_state=42)

# # =========================
# # PAIR GENERATOR (IMPORTANT FIX)
# # =========================
# def pair_generator(people_list, batch_size=32):
#     while True:
#         X1, X2, y = [], [], []

#         for _ in range(batch_size):
#             p1 = random.choice(people_list)
#             img1 = random.choice(data[p1])

#             if random.random() < 0.5:
#                 # positive pair
#                 img2 = random.choice(data[p1])
#                 label = 1
#             else:
#                 # negative pair
#                 p2 = random.choice(people_list)
#                 while p2 == p1:
#                     p2 = random.choice(people_list)
#                 img2 = random.choice(data[p2])
#                 label = 0

#             X1.append(img1)
#             X2.append(img2)
#             y.append(label)

#         yield (np.array(X1), np.array(X2)), np.array(y)

# # =========================
# # BASE MODEL (IMPROVED ACCURACY)
# # =========================
# def build_base_network():
#     base = tf.keras.applications.MobileNetV2(
#         input_shape=(IMG_SIZE, IMG_SIZE, 3),
#         include_top=False,
#         weights="imagenet",
#         pooling="avg"
#     )

#     for layer in base.layers[:-30]:
#         layer.trainable = False

#     input = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
#     x = base(input)
#     x = layers.Dense(128, activation="relu")(x)
#     x = layers.Dropout(0.3)(x)
#     x = layers.Dense(64)(x)

#     # L2 normalize (VERY IMPORTANT for Siamese)
#     output = layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=1))(x)

#     return Model(input, output)

# base_network = build_base_network()

# # =========================
# # SIAMESE MODEL (NO Lambda distance issues)
# # =========================
# input_a = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
# input_b = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

# emb_a = base_network(input_a)
# emb_b = base_network(input_b)

# # L1 distance as simple layer (SAFE)
# distance = layers.Subtract()([emb_a, emb_b])
# distance = layers.Lambda(lambda x: tf.abs(x))(distance)

# x = layers.Dense(32, activation="relu")(distance)
# output = layers.Dense(1, activation="sigmoid")(x)

# model = Model([input_a, input_b], output)

# model.compile(
#     optimizer=tf.keras.optimizers.Adam(1e-4),
#     loss="binary_crossentropy",
#     metrics=["accuracy"]
# )

# model.summary()

# # =========================
# # TRAINING
# # =========================
# print("\nTraining Started...\n")

# train_steps = 200
# test_steps = 50

# history = model.fit(
#     pair_generator(train_people, 32),
#     steps_per_epoch=train_steps,
#     validation_data=pair_generator(test_people, 32),
#     validation_steps=test_steps,
#     epochs=15,
#     callbacks=[
#         tf.keras.callbacks.ModelCheckpoint(
#             "face_siamese.keras",
#             save_best_only=True
#         ),
#         tf.keras.callbacks.ReduceLROnPlateau(
#             monitor="val_loss",
#             factor=0.5,
#             patience=2
#         ),
#         tf.keras.callbacks.EarlyStopping(
#             patience=4,
#             restore_best_weights=True
#         )
#     ]
# )

# # =========================
# # SAVE FINAL MODEL
# # =========================
# model.save("face_siamese.keras")
# print("\nModel Saved ✔")















# import numpy as np   thoda crash hua ...thoda chla
# import tensorflow as tf

# from tensorflow.keras.layers import (
#     Input,
#     Dense,
#     Lambda,
#     Dropout,
#     BatchNormalization
# )

# from tensorflow.keras.models import Model
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import (
#     EarlyStopping,
#     ModelCheckpoint
# )

# from tensorflow.keras.applications import MobileNetV2
# from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# # =====================================================
# # LOAD DATASET
# # =====================================================

# print("Loading Dataset...")

# X_train = np.load(
#     "dataset/X_train.npy",
#     mmap_mode="r"
# )

# X_test = np.load(
#     "dataset/X_test.npy",
#     mmap_mode="r"
# )

# y_train = np.load("dataset/y_train.npy")
# y_test = np.load("dataset/y_test.npy")

# print("Train Shape:", X_train.shape)
# print("Test Shape :", X_test.shape)

# IMG_SIZE = X_train.shape[2]

# # =====================================================
# # PREPROCESS FUNCTION
# # =====================================================

# def preprocess_images(images):

#     images = images.astype("float32")

#     images = preprocess_input(images)

#     return images


# # =====================================================
# # BUILD EMBEDDING NETWORK
# # =====================================================

# def build_embedding_network():

#     base_model = MobileNetV2(
#         weights="imagenet",
#         include_top=False,
#         input_shape=(IMG_SIZE, IMG_SIZE, 3),
#         pooling="avg"
#     )

#     # Freeze initial layers
#     for layer in base_model.layers[:-20]:
#         layer.trainable = False

#     input_layer = Input(
#         shape=(IMG_SIZE, IMG_SIZE, 3)
#     )

#     x = base_model(input_layer)

#     x = Dense(
#         256,
#         activation="relu"
#     )(x)

#     x = BatchNormalization()(x)

#     x = Dropout(0.3)(x)

#     x = Dense(
#         128,
#         activation="relu"
#     )(x)

#     return Model(
#         input_layer,
#         x,
#         name="EmbeddingNetwork"
#     )


# embedding_network = build_embedding_network()

# # =====================================================
# # SIAMESE INPUTS
# # =====================================================

# input_a = Input(
#     shape=(IMG_SIZE, IMG_SIZE, 3)
# )

# input_b = Input(
#     shape=(IMG_SIZE, IMG_SIZE, 3)
# )

# embedding_a = embedding_network(input_a)
# embedding_b = embedding_network(input_b)

# # =====================================================
# # L1 DISTANCE
# # =====================================================

# def l1_distance(vectors):

#     x, y = vectors

#     return tf.abs(x - y)


# distance = Lambda(
#     l1_distance,
#     name="L1_Distance"
# )([embedding_a, embedding_b])

# # =====================================================
# # OUTPUT
# # =====================================================

# output = Dense(
#     1,
#     activation="sigmoid"
# )(distance)

# # =====================================================
# # CREATE MODEL
# # =====================================================

# model = Model(
#     inputs=[input_a, input_b],
#     outputs=output,
#     name="SiameseNetwork"
# )

# # =====================================================
# # COMPILE
# # =====================================================

# model.compile(
#     optimizer=Adam(
#         learning_rate=0.0001
#     ),
#     loss="binary_crossentropy",
#     metrics=["accuracy"]
# )

# model.summary()

# # =====================================================
# # PREPROCESS DATA
# # =====================================================

# print("\nPreprocessing Images...\n")

# X1_train = preprocess_images(
#     np.array(X_train[:, 0])
# )

# X2_train = preprocess_images(
#     np.array(X_train[:, 1])
# )

# X1_test = preprocess_images(
#     np.array(X_test[:, 0])
# )

# X2_test = preprocess_images(
#     np.array(X_test[:, 1])
# )

# # =====================================================
# # CALLBACKS
# # =====================================================

# # early_stop = EarlyStopping(
# #     monitor="val_accuracy",
# #     patience=3,
# #     restore_best_weights=True,
# #     verbose=1
# # ) replacing this with below line 

# early_stop = EarlyStopping(
#     monitor="val_accuracy",
#     patience=2,
#     restore_best_weights=True,
#     verbose=1
# )

# checkpoint = ModelCheckpoint(
#     "face_recognition_model.keras",
#     monitor="val_accuracy",
#     save_best_only=True,
#     verbose=1
# )

# # =====================================================
# # TRAIN
# # =====================================================

# print("\nTraining Started...\n")

# history = model.fit(
#     [X1_train, X2_train],
#     y_train,
#     validation_data=(
#         [X1_test, X2_test],
#         y_test
#     ),
#     batch_size=8,
#     epochs=5,
#     callbacks=[
#         early_stop,
#         checkpoint
#     ]
# )

# # =====================================================
# # EVALUATE
# # =====================================================

# print("\nEvaluating Model...\n")

# loss, accuracy = model.evaluate(
#     [X1_test, X2_test],
#     y_test
# )

# print("\nTest Loss :", loss)
# print("Test Accuracy :", accuracy)

# # =====================================================
# # SAVE
# # =====================================================

# model.save(
#     "face_recognition_model.keras"
# )

# print("\nModel Saved Successfully!")


















# import numpy as np             yeh without changes
# import tensorflow as tf

# from tensorflow.keras.layers import (
#     Input,
#     Conv2D,
#     MaxPooling2D,
#     Flatten,
#     Dense,
#     Lambda,
#     BatchNormalization,
#     Dropout
# )

# from tensorflow.keras.models import Model
# from tensorflow.keras.optimizers import Adam

# from tensorflow.keras.callbacks import (
#     EarlyStopping,
#     ModelCheckpoint
# )

# # =====================================================
# # LOAD DATASET
# # =====================================================

# print("Loading training dataset...")

# X_train = np.load("dataset/X_train.npy")
# X_test = np.load("dataset/X_test.npy")

# y_train = np.load("dataset/y_train.npy")
# y_test = np.load("dataset/y_test.npy")

# print("Training pairs:", len(X_train))
# print("Testing pairs:", len(X_test))
# print("Dataset shape:", X_train.shape)

# # =====================================================
# # EMBEDDING NETWORK
# # =====================================================

# def build_embedding_network():

#     input_layer = Input(shape=(100, 100, 3))

#     x = Conv2D(
#         64,
#         (3, 3),
#         activation='relu',
#         padding='same'
#     )(input_layer)

#     x = BatchNormalization()(x)
#     x = MaxPooling2D()(x)

#     x = Conv2D(
#         128,
#         (3, 3),
#         activation='relu',
#         padding='same'
#     )(x)

#     x = BatchNormalization()(x)
#     x = MaxPooling2D()(x)

#     x = Conv2D(
#         128,
#         (3, 3),
#         activation='relu',
#         padding='same'
#     )(x)

#     x = BatchNormalization()(x)
#     x = MaxPooling2D()(x)

#     x = Flatten()(x)

#     x = Dense(
#         512,
#         activation='relu'
#     )(x)

#     # Reduce Overfitting
#     x = Dropout(0.5)(x)

#     return Model(
#         input_layer,
#         x,
#         name="EmbeddingNetwork"
#     )

# # Build embedding model
# embedding_network = build_embedding_network()

# # =====================================================
# # SIAMESE NETWORK INPUTS
# # =====================================================

# input_a = Input(shape=(100, 100, 3))
# input_b = Input(shape=(100, 100, 3))

# feature_a = embedding_network(input_a)
# feature_b = embedding_network(input_b)

# # =====================================================
# # L1 DISTANCE FUNCTION
# # =====================================================

# def l1_distance(tensors):
#     x, y = tensors
#     return tf.abs(x - y)

# distance = Lambda(
#     l1_distance,
#     output_shape=(512,),
#     name="L1_Distance"
# )([feature_a, feature_b])

# # =====================================================
# # OUTPUT LAYER
# # =====================================================

# output = Dense(
#     1,
#     activation='sigmoid'
# )(distance)

# # =====================================================
# # CREATE MODEL
# # =====================================================

# model = Model(
#     inputs=[input_a, input_b],
#     outputs=output,
#     name="SiameseNetwork"
# )

# # =====================================================
# # COMPILE MODEL
# # =====================================================

# model.compile(
#     optimizer=Adam(
#         learning_rate=0.0001
#     ),
#     loss='binary_crossentropy',
#     metrics=['accuracy']
# )

# model.summary()

# # =====================================================
# # CALLBACKS
# # =====================================================

# early_stop = EarlyStopping(
#     monitor='val_accuracy',
#     patience=2,
#     restore_best_weights=True,
#     verbose=1
# )

# checkpoint = ModelCheckpoint(
#     "face_recognition_model.keras",
#     monitor='val_accuracy',
#     save_best_only=True,
#     verbose=1
# )

# # =====================================================
# # TRAIN MODEL
# # =====================================================

# print("\nTraining model...\n")

# history = model.fit(
#     [X_train[:, 0], X_train[:, 1]],
#     y_train,
#     validation_data=(
#         [X_test[:, 0], X_test[:, 1]],
#         y_test
#     ),
#     batch_size=32,
#     epochs=10,
#     callbacks=[
#         early_stop,
#         checkpoint
#     ]
# )

# # =====================================================
# # EVALUATE MODEL
# # =====================================================

# print("\nEvaluating model...\n")

# loss, accuracy = model.evaluate(
#     [X_test[:, 0], X_test[:, 1]],
#     y_test
# )

# print("\nTest Loss:", loss)
# print("Test Accuracy:", accuracy)

# # =====================================================
# # SAVE MODEL
# # =====================================================

# model.save("face_recognition_model.keras")

# print("\nModel saved successfully!")















# import numpy as np
# import tensorflow as tf

# from tensorflow.keras.layers import (
#     Input,
#     Conv2D,
#     MaxPooling2D,
#     Flatten,
#     Dense,
#     Lambda
# )

# from tensorflow.keras.models import Model
# from tensorflow.keras.optimizers import Adam

# # -----------------------------
# # Load Dataset
# # -----------------------------

# print("Loading training dataset...")

# X_train = np.load("dataset/X_train.npy")
# X_test = np.load("dataset/X_test.npy")
# y_train = np.load("dataset/y_train.npy")
# y_test = np.load("dataset/y_test.npy")

# print("Training pairs:", len(X_train))
# print("Testing pairs:", len(X_test))

# print("Dataset shape:", X_train.shape)

# # -----------------------------
# # Build Embedding Network
# # -----------------------------

# def build_embedding_network():

#     input_layer = Input(shape=(100,100,3))

#     x = Conv2D(64,(3,3),activation='relu')(input_layer)
#     x = MaxPooling2D()(x)

#     x = Conv2D(128,(3,3),activation='relu')(x)
#     x = MaxPooling2D()(x)

#     x = Conv2D(128,(3,3),activation='relu')(x)
#     x = MaxPooling2D()(x)

#     x = Flatten()(x)

#     x = Dense(512,activation='relu')(x)

#     return Model(input_layer,x)

# embedding_network = build_embedding_network()

# # -----------------------------
# # Siamese Inputs
# # -----------------------------

# input_a = Input(shape=(100,100,3))
# input_b = Input(shape=(100,100,3))

# feature_a = embedding_network(input_a)
# feature_b = embedding_network(input_b)

# # -----------------------------
# # L1 Distance Layer
# # -----------------------------

# # distance = Lambda(
# #     lambda tensors: tf.abs(tensors[0] - tensors[1]),
# #     output_shape=(512,)
# # )([feature_a, feature_b])    replacing this

# def l1_distance(tensors):
#     x, y = tensors
#     return tf.abs(x - y)

# distance = Lambda(
#     l1_distance,
#     output_shape=(512,)
# )([feature_a, feature_b])

# # -----------------------------
# # Output Layer
# # -----------------------------

# output = Dense(1,activation='sigmoid')(distance)

# model = Model(
#     inputs=[input_a,input_b],
#     outputs=output
# )

# # -----------------------------
# # Compile Model
# # -----------------------------

# model.compile(
#     loss='binary_crossentropy',
#     optimizer=Adam(learning_rate=0.0001),
#     metrics=['accuracy']
# )

# model.summary()

# # -----------------------------
# # Train
# # -----------------------------

# print("Training model...")

# history = model.fit(
#     [X_train[:,0], X_train[:,1]],
#     y_train,
#     validation_data=(
#         [X_test[:,0], X_test[:,1]],
#         y_test
#     ),
#     batch_size=32,
#     epochs=15
# )

# # -----------------------------
# # Evaluate
# # -----------------------------

# print("\nEvaluating model...")

# loss, accuracy = model.evaluate(
#     [X_test[:,0], X_test[:,1]],
#     y_test
# )

# print("Test Loss:", loss)
# print("Test Accuracy:", accuracy)

# # -----------------------------
# # Save Model
# # -----------------------------

# model.save("face_recognition_model.keras")

# print("Model saved successfully!")








# import numpy as np           aaj ka 
# import tensorflow as tf
# from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input, Lambda, MaxPooling2D
# from tensorflow.keras.models import Model
# from tensorflow.keras.optimizers import Adam

# # -----------------------------
# # Load Dataset
# # -----------------------------

# print("Loading training dataset...")

# X_train = np.load("dataset/X_train.npy")
# X_test = np.load("dataset/X_test.npy")
# y_train = np.load("dataset/y_train.npy")
# y_test = np.load("dataset/y_test.npy")

# print("Training pairs:", len(X_train))
# print("Testing pairs:", len(X_test))


# # -----------------------------
# # Build Embedding Network
# # -----------------------------


# def build_embedding_network():
#     input_layer = Input(shape=(100, 100, 1))

#     x = Conv2D(64, (3, 3), activation="relu")(input_layer)
#     x = MaxPooling2D()(x)

#     x = Conv2D(128, (3, 3), activation="relu")(x)
#     x = MaxPooling2D()(x)

#     x = Conv2D(128, (3, 3), activation="relu")(x)
#     x = MaxPooling2D()(x)

#     x = Flatten()(x)
#     x = Dense(512, activation="relu")(x)

#     return Model(input_layer, x)


# embedding_network = build_embedding_network()

# # -----------------------------
# # Siamese Network Inputs
# # -----------------------------

# input_a = Input(shape=(100, 100, 1))
# input_b = Input(shape=(100, 100, 1))

# feature_a = embedding_network(input_a)
# feature_b = embedding_network(input_b)

# # -----------------------------
# # L1 Distance Layer
# # -----------------------------

# distance = Lambda(
#     lambda tensors: tf.abs(tensors[0] - tensors[1]), output_shape=(512,)
# )([feature_a, feature_b])

# output = Dense(1, activation="sigmoid")(distance)

# model = Model(inputs=[input_a, input_b], outputs=output)

# # -----------------------------
# # Compile Model
# # -----------------------------

# model.compile(
#     loss="binary_crossentropy",
#     optimizer=Adam(learning_rate=0.0001),
#     metrics=["accuracy"],
# )

# model.summary()

# # -----------------------------
# # Train Model
# # -----------------------------

# print("Training model...")

# history = model.fit(
#     [X_train[:, 0], X_train[:, 1]],
#     y_train,
#     validation_data=([X_test[:, 0], X_test[:, 1]], y_test),
#     batch_size=32,
#     epochs=10,
# )

# # -----------------------------
# # Save Model
# # -----------------------------

# model.save("face_recognition_model.keras")
# print("Model saved successfully!")

# # -----------------------------
# # Evaluate Model
# # -----------------------------

# loss, accuracy = model.evaluate([X_test[:, 0], X_test[:, 1]], y_test)
# print("Test Accuracy:", accuracy)






# import numpy as np
# import tensorflow as tf

# from tensorflow.keras.layers import Input, Lambda, Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
# from tensorflow.keras.models import Model
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.applications import MobileNetV2

# # -----------------------------
# # Load Dataset with Memory Mapping
# # -----------------------------

# print("Loading dataset memory maps...")

# # Use mmap_mode='r' to prevent high memory consumption and swapping
# X_train_mmap = np.load("dataset/X_train.npy", mmap_mode="r")
# X_test_mmap = np.load("dataset/X_test.npy", mmap_mode="r")
# y_train = np.load("dataset/y_train.npy").astype("float32")
# y_test = np.load("dataset/y_test.npy").astype("float32")

# print("Training pairs:", len(X_train_mmap))
# print("Testing pairs:", len(X_test_mmap))

# # Load validation dataset to memory (only 3471 pairs, ~800MB in float32, perfectly safe)
# print("Loading validation set...")
# X1_test = X_test_mmap[:, 0].astype("float32") / 255.0
# X2_test = X_test_mmap[:, 1].astype("float32") / 255.0

# # -----------------------------
# # Data Generator using Sequence (Robust & memory-safe)
# # -----------------------------

# from tensorflow.keras.utils import Sequence

# class SiameseGenerator(Sequence):
#     def __init__(self, X, y, batch_size=32, shuffle=True, **kwargs):
#         super().__init__(**kwargs)
#         self.X = X
#         self.y = y
#         self.batch_size = batch_size
#         self.shuffle = shuffle
#         self.indices = np.arange(len(self.X))
#         self.on_epoch_end()

#     def __len__(self):
#         return int(np.floor(len(self.X) / self.batch_size))

#     def __getitem__(self, idx):
#         start = idx * self.batch_size
#         end = start + self.batch_size
#         batch_idx = self.indices[start:end]

#         # Extract batch and normalize on the fly
#         img1 = self.X[batch_idx, 0].astype("float32") / 255.0
#         img2 = self.X[batch_idx, 1].astype("float32") / 255.0
#         batch_y = self.y[batch_idx]

#         # Return a tuple instead of a list to prevent Keras 3 TypeSpec signature bugs
#         return (img1, img2), batch_y

#     def on_epoch_end(self):
#         if self.shuffle:
#             np.random.shuffle(self.indices)

# # -----------------------------
# # Build Embedding Network (MobileNetV2 Transfer Learning)
# # -----------------------------

# def build_embedding_network():

#     # MobileNetV2 expects 3 channels
#     base_model = MobileNetV2(
#         input_shape=(100, 100, 3),
#         include_top=False,
#         weights='imagenet'
#     )

#     # Freeze base model layers to prevent overfitting and speed up CPU training
#     base_model.trainable = False

#     input_layer = Input(shape=(100, 100, 3))
#     x = base_model(input_layer, training=False)
#     x = GlobalAveragePooling2D()(x)
#     x = Dense(256, activation='relu')(x)
#     x = BatchNormalization()(x)
#     x = Dropout(0.3)(x)
#     x = Dense(128, activation='linear')(x)
#     # L2 normalize embeddings so that distance metrics are robust and bounded
#     x = Lambda(lambda t: tf.math.l2_normalize(t, axis=1))(x)

#     return Model(input_layer, x)


# embedding_network = build_embedding_network()

# # -----------------------------
# # Siamese Network Inputs
# # -----------------------------

# input_a = Input(shape=(100, 100, 3))
# input_b = Input(shape=(100, 100, 3))

# feature_a = embedding_network(input_a)
# feature_b = embedding_network(input_b)

# # -----------------------------
# # L1 Distance Layer (FIXED)
# # -----------------------------

# distance = Lambda(
#     lambda tensors: abs(tensors[0] - tensors[1]),
#     output_shape=(128,)
# )([feature_a, feature_b])

# output = Dense(1, activation='sigmoid')(distance)

# model = Model(inputs=[input_a, input_b], outputs=output)

# # -----------------------------
# # Compile Model
# # -----------------------------

# model.compile(
#     loss='binary_crossentropy',
#     optimizer=Adam(learning_rate=0.0002),
#     metrics=['accuracy']
# )

# model.summary()

# # -----------------------------
# # Train Model
# # -----------------------------

# print("Training model...")

# batch_size = 32
# epochs = 15

# train_gen = SiameseGenerator(X_train_mmap, y_train, batch_size=batch_size, shuffle=True)

# # Train using Sequence generator to preserve memory and avoid TypeSpec errors
# history = model.fit(
#     train_gen,
#     validation_data=((X1_test, X2_test), y_test),
#     epochs=epochs
# )

# # -----------------------------
# # Save Model
# # -----------------------------

# model.save("face_recognition_model.keras")

# print("Model saved successfully!")

# loss, accuracy = model.evaluate((X1_test, X2_test), y_test)

# print("Test Accuracy:", accuracy)
# print('Loss:', loss)




# YEH SABSE PEHLE WAALE


# import numpy as np
# import tensorflow as tf

# from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D
# from tensorflow.keras.layers import Flatten, Dense, Lambda
# from tensorflow.keras.models import Model
# from tensorflow.keras.optimizers import Adam

# # -----------------------------
# # Load dataset
# # -----------------------------

# X_train = np.load("X_train.npy")
# X_test = np.load("X_test.npy")
# y_train = np.load("y_train.npy")
# y_test = np.load("y_test.npy")

# print("Training pairs:", len(X_train))
# print("Testing pairs:", len(X_test))

# # -----------------------------
# # Embedding network
# # -----------------------------

# def build_embedding():

#     inp = Input(shape=(100,100,1))

#     x = Conv2D(64,(3,3),activation='relu')(inp)
#     x = MaxPooling2D()(x)

#     x = Conv2D(128,(3,3),activation='relu')(x)
#     x = MaxPooling2D()(x)

#     x = Conv2D(128,(3,3),activation='relu')(x)
#     x = MaxPooling2D()(x)

#     x = Flatten()(x)

#     x = Dense(512,activation='relu')(x)

#     return Model(inp,x)

# embedding = build_embedding()

# # -----------------------------
# # Siamese Inputs
# # -----------------------------

# input_a = Input(shape=(100,100,1))
# input_b = Input(shape=(100,100,1))

# feat_a = embedding(input_a)
# feat_b = embedding(input_b)

# # FIXED PART
# distance = Lambda(lambda tensors: tf.abs(tensors[0] - tensors[1]))([feat_a, feat_b])

# output = Dense(1,activation='sigmoid')(distance)

# model = Model(inputs=[input_a,input_b],outputs=output)

# # -----------------------------
# # Compile
# # -----------------------------

# model.compile(
#     loss='binary_crossentropy',
#     optimizer=Adam(0.0001),
#     metrics=['accuracy']
# )

# model.summary()

# # -----------------------------
# # Train
# # -----------------------------

# history = model.fit(
#     [X_train[:,0], X_train[:,1]],
#     y_train,
#     validation_data=([X_test[:,0],X_test[:,1]],y_test),
#     batch_size=16,
#     epochs=10
# )

# # -----------------------------
# # Save model
# # -----------------------------

# model.save("face_recognition_model.h5")

# print("Model saved successfully")










# import numpy as np
# import tensorflow as tf
# from tensorflow.keras import layers, Model


# IMG_SIZE = 100


# def build_embedding():

#     input = layers.Input((IMG_SIZE, IMG_SIZE, 1))

#     x = layers.Conv2D(64, (3,3), activation='relu')(input)
#     x = layers.MaxPooling2D()(x)

#     x = layers.Conv2D(128, (3,3), activation='relu')(x)
#     x = layers.MaxPooling2D()(x)

#     x = layers.Conv2D(128, (3,3), activation='relu')(x)
#     x = layers.MaxPooling2D()(x)

#     x = layers.Flatten()(x)

#     x = layers.Dense(256, activation='relu')(x)

#     return Model(input, x)


# embedding = build_embedding()

# input_a = layers.Input((IMG_SIZE, IMG_SIZE, 1))
# input_b = layers.Input((IMG_SIZE, IMG_SIZE, 1))

# feat_a = embedding(input_a)
# feat_b = embedding(input_b)

# from tensorflow.keras.layers import Lambda

# distance = Lambda(lambda tensors: tf.abs(tensors[0] - tensors[1]))([feat_a, feat_b])

# output = layers.Dense(1, activation='sigmoid')(distance)

# model = Model([input_a, input_b], output)

# model.compile(
#     loss='binary_crossentropy',
#     optimizer='adam',
#     metrics=['accuracy']
# )

# print(model.summary())

# print("Loading dataset...")

# X_train = np.load("dataset/X_train.npy")
# X_test = np.load("dataset/X_test.npy")
# y_train = np.load("dataset/y_train.npy")
# y_test = np.load("dataset/y_test.npy")

# X1_train = X_train[:,0]
# X2_train = X_train[:,1]

# X1_test = X_test[:,0]
# X2_test = X_test[:,1]

# print("Training model...")

# model.fit(
#     [X1_train, X2_train],
#     y_train,
#     validation_data=([X1_test, X2_test], y_test),
#     epochs=10,
#     batch_size=32
# )

# model.save("face_recognition_model.h5")

# print("Model saved successfully")