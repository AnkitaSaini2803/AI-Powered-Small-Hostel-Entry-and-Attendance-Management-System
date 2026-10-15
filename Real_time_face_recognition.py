import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# =====================================
# CONFIG
# =====================================
IMG_SIZE = 96
SIMILARITY_THRESHOLD = 0.75

# =====================================
# LOAD MODEL
# =====================================
embedding_model = load_model("embedding_model.keras", compile=False)
embeddings_db = np.load("embeddings.npy")
labels_db = np.load("labels.npy")

# =====================================
# FACE DETECTOR
# =====================================
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# =====================================
# COSINE SIMILARITY
# =====================================
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# =====================================
# PREPROCESS
# =====================================
def preprocess_face(face):
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
    face = face.astype("float32")
    face = preprocess_input(face)
    return face

# =====================================
# START CAMERA
# =====================================
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]

        try:
            processed = preprocess_face(face)
            embedding = embedding_model.predict(
                np.expand_dims(processed, axis=0),
                verbose=0
            )[0]

            best_score = -1
            best_label = "Unknown"

            for db_embedding, db_label in zip(embeddings_db, labels_db):
                score = cosine_similarity(embedding, db_embedding)
                if score > best_score:
                    best_score = score
                    best_label = db_label

            if best_score < SIMILARITY_THRESHOLD:
                best_label = "Unknown"
                color = (0, 0, 255)
            else:
                color = (0, 255, 0)

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(
                frame,
                f"{best_label} {best_score:.2f}",
                (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

        except Exception as e:
            print(e)

    cv2.imshow("Hostel Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()






# import cv2
# import numpy as np

# from tensorflow.keras.models import load_model
# from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# # =====================================
# # CONFIG
# # =====================================

# IMG_SIZE = 96

# SIMILARITY_THRESHOLD = 0.75

# # =====================================
# # LOAD MODEL
# # =====================================

# embedding_model = load_model(
#     "embedding_model.keras",
#     compile=False
# )

# embeddings_db = np.load(
#     "embeddings.npy"
# )

# labels_db = np.load(
#     "labels.npy"
# )

# # =====================================
# # FACE DETECTOR
# # =====================================

# face_detector = cv2.CascadeClassifier(
#     cv2.data.haarcascades +
#     "haarcascade_frontalface_default.xml"
# )

# # =====================================
# # COSINE SIMILARITY
# # =====================================

# def cosine_similarity(a, b):

#     return np.dot(a, b) / (
#         np.linalg.norm(a) *
#         np.linalg.norm(b)
#     )

# # =====================================
# # PREPROCESS
# # =====================================

# def preprocess_face(face):

#     face = cv2.cvtColor(
#         face,
#         cv2.COLOR_BGR2RGB
#     )

#     face = cv2.resize(
#         face,
#         (IMG_SIZE, IMG_SIZE)
#     )

#     face = face.astype("float32")

#     face = preprocess_input(face)

#     return face

# # =====================================
# # START CAMERA
# # =====================================

# cap = cv2.VideoCapture(0)

# while True:

#     ret, frame = cap.read()

#     if not ret:
#         break

#     gray = cv2.cvtColor(
#         frame,
#         cv2.COLOR_BGR2GRAY
#     )

#     faces = face_detector.detectMultiScale(
#         gray,
#         scaleFactor=1.1,
#         minNeighbors=5
#     )

#     for (x, y, w, h) in faces:

#         face = frame[
#             y:y+h,
#             x:x+w
#         ]

#         try:

#             processed = preprocess_face(
#                 face
#             )

#             embedding = embedding_model.predict(
#                 np.expand_dims(processed, axis=0),
#                 verbose=0
#             )[0]

#             best_score = -1
#             best_label = "Unknown"

#             for db_embedding, db_label in zip(
#                 embeddings_db,
#                 labels_db
#             ):

#                 score = cosine_similarity(
#                     embedding,
#                     db_embedding
#                 )

#                 if score > best_score:

#                     best_score = score
#                     best_label = db_label

#             if best_score < SIMILARITY_THRESHOLD:

#                 best_label = "Unknown"
#                 color = (0, 0, 255)

#             else:

#                 color = (0, 255, 0)

#             cv2.rectangle(
#                 frame,
#                 (x, y),
#                 (x+w, y+h),
#                 color,
#                 2
#             )

#             cv2.putText(
#                 frame,
#                 f"{best_label} {best_score:.2f}",
#                 (x, y-10),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.8,
#                 color,
#                 2
#             )

#         except Exception as e:
#             print(e)

#     cv2.imshow(
#         "Hostel Face Recognition",
#         frame
#     )

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()




# import cv2
# import numpy as np
# import tensorflow as tf

# from tensorflow.keras.models import load_model
# from tensorflow.keras.applications.mobilenet_v2 import (
#     preprocess_input
# )

# # =====================================
# # CONFIGURATION
# # =====================================

# IMG_SIZE = 96
# THRESHOLD = 0.36

# # =====================================
# # CUSTOM LAYER
# # =====================================

# def l1_distance(tensors):

#     x, y = tensors

#     return tf.abs(x - y)

# # =====================================
# # LOAD MODEL
# # =====================================

# print("Loading Model...")

# model = load_model(
#     "face_recognition_model.keras",
#     compile=False,
#     custom_objects={
#         "l1_distance": l1_distance
#     }
# )

# print("Model Loaded Successfully!")

# # =====================================
# # FACE DETECTOR
# # =====================================

# face_cascade = cv2.CascadeClassifier(
#     cv2.data.haarcascades +
#     "haarcascade_frontalface_default.xml"
# )

# # =====================================
# # CAMERA
# # =====================================

# cap = cv2.VideoCapture(0)

# if not cap.isOpened():

#     print("Could not open webcam.")
#     exit()

# reference_face = None

# print("\n==============================")
# print("Press S -> Save Reference Face")
# print("Press Q -> Quit")
# print("==============================\n")

# # =====================================
# # MAIN LOOP
# # =====================================

# while True:

#     ret, frame = cap.read()

#     if not ret:
#         break

#     gray = cv2.cvtColor(
#         frame,
#         cv2.COLOR_BGR2GRAY
#     )

#     faces = face_cascade.detectMultiScale(
#         gray,
#         scaleFactor=1.1,
#         minNeighbors=5,
#         minSize=(80, 80)
#     )

#     for (x, y, w, h) in faces:

#         # -------------------------
#         # Crop Face
#         # -------------------------

#         face = frame[
#             y:y+h,
#             x:x+w
#         ]

#         face = cv2.resize(
#             face,
#             (IMG_SIZE, IMG_SIZE)
#         )

#         # -------------------------
#         # MobileNetV2 Preprocessing
#         # -------------------------

#         face = face.astype(
#             "float32"
#         )

#         face = preprocess_input(
#             face
#         )

#         face = np.expand_dims(
#             face,
#             axis=0
#         )

#         # -------------------------
#         # Compare
#         # -------------------------

#         if reference_face is not None:

#             score = model.predict(
#                 [reference_face, face],
#                 verbose=0
#             )[0][0]
#             # Added this 
#             print("Score =", score)

#             if score >= THRESHOLD:

#                 label = (
#                     f"MATCH "
#                     f"{score*100:.1f}%"
#                 )

#                 color = (
#                     0,
#                     255,
#                     0
#                 )

#             else:

#                 label = (
#                     f"NOT MATCH "
#                     f"{score*100:.1f}%"
#                 )

#                 color = (
#                     0,
#                     0,
#                     255
#                 )

#             cv2.putText(
#                 frame,
#                 label,
#                 (x, y - 10),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.7,
#                 color,
#                 2
#             )

#         # -------------------------
#         # Draw Rectangle
#         # -------------------------

#         cv2.rectangle(
#             frame,
#             (x, y),
#             (x+w, y+h),
#             (255, 0, 0),
#             2
#         )

#     # ---------------------------------
#     # Show Frame
#     # ---------------------------------

#     cv2.imshow(
#         "Face Verification",
#         frame
#     )

#     key = cv2.waitKey(1) & 0xFF

#     # ---------------------------------
#     # Save Reference Face
#     # ---------------------------------

#     if key == ord('s'):

#         if len(faces) > 0:

#             (x, y, w, h) = faces[0]

#             ref = frame[
#                 y:y+h,
#                 x:x+w
#             ]

#             ref = cv2.resize(
#                 ref,
#                 (IMG_SIZE, IMG_SIZE)
#             )

#             ref = ref.astype(
#                 "float32"
#             )

#             ref = preprocess_input(
#                 ref
#             )

#             reference_face = np.expand_dims(
#                 ref,
#                 axis=0
#             )

#             print(
#                 "Reference Face Saved!"
#             )

#         else:

#             print(
#                 "No face detected."
#             )

#     # ---------------------------------
#     # Quit
#     # ---------------------------------

#     elif key == ord('q'):

#         break

# # =====================================
# # CLEANUP
# # =====================================

# cap.release()

# cv2.destroyAllWindows()

# print("Program Closed.")






# import cv2
# import numpy as np
# import tensorflow as tf

# from tensorflow.keras.models import load_model

# IMG_SIZE = 96

# # =====================================
# # CUSTOM LAYER
# # =====================================

# def l1_distance(tensors):
#     x,y = tensors
#     return tf.abs(x-y)

# # =====================================
# # LOAD MODEL
# # =====================================

# model = load_model(
#     "face_recognition_model.keras",
#     compile=False,
#     custom_objects={
#         "l1_distance": l1_distance
#     }
# )

# # =====================================
# # FACE DETECTOR
# # =====================================

# face_cascade = cv2.CascadeClassifier(
#     cv2.data.haarcascades +
#     "haarcascade_frontalface_default.xml"
# )

# cap = cv2.VideoCapture(0)

# reference_face = None

# print("Press S to capture reference face")
# print("Press Q to quit")

# while True:

#     ret, frame = cap.read()

#     if not ret:
#         break

#     gray = cv2.cvtColor(
#         frame,
#         cv2.COLOR_BGR2GRAY
#     )

#     faces = face_cascade.detectMultiScale(
#         gray,
#         1.3,
#         5
#     )

#     for (x,y,w,h) in faces:

#         face = frame[y:y+h,x:x+w]

#         face = cv2.resize(
#             face,
#             (IMG_SIZE,IMG_SIZE)
#         )

#         face = face.astype("float32")/255.0

#         face = np.expand_dims(
#             face,
#             axis=0
#         )

#         if reference_face is not None:

#             score = model.predict(
#                 [reference_face,face],
#                 verbose=0
#             )[0][0]

#             if score > 0.32:

#                 label = f"MATCH {score:.2f}"
#                 color=(0,255,0)

#             else:

#                 label = f"NOT MATCH {score:.2f}"
#                 color=(0,0,255)

#             cv2.putText(
#                 frame,
#                 label,
#                 (x,y-10),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.8,
#                 color,
#                 2
#             )

#         cv2.rectangle(
#             frame,
#             (x,y),
#             (x+w,y+h),
#             (255,0,0),
#             2
#         )

#     cv2.imshow(
#         "Face Verification",
#         frame
#     )

#     key = cv2.waitKey(1)

#     if key == ord('s'):

#         if len(faces) > 0:

#             (x,y,w,h) = faces[0]

#             ref = frame[y:y+h,x:x+w]

#             ref = cv2.resize(
#                 ref,
#                 (IMG_SIZE,IMG_SIZE)
#             )

#             ref = ref.astype("float32")/255.0

#             reference_face = np.expand_dims(
#                 ref,
#                 axis=0
#             )

#             print("Reference face saved")

#     elif key == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()









# import cv2         yeh without changes  5-6-2026
# import numpy as np
# # import keras
# import tensorflow as tf

# from tensorflow.keras.models import load_model

# # keras.config.enable_unsafe_deserialization()

# IMG_SIZE = 100
# def l1_distance(tensors):
#     x, y = tensors
#     return tf.abs(x - y)
# # Load Model
# model = load_model(
#     "face_recognition_model.keras",
#     compile=False,
#     custom_objects={
#         "l1_distance": l1_distance
#     }
# )
# # model = load_model(
# #     "face_recognition_model.keras",
# #     compile=False,
# #     # new line
# #     custom_objects={"tf": tf}
# # )

# # Face Detector
# face_cascade = cv2.CascadeClassifier(
#     cv2.data.haarcascades +
#     "haarcascade_frontalface_default.xml"
# )

# cap = cv2.VideoCapture(0)

# print("Press S to capture reference face")
# print("Press Q to quit")

# reference_face = None

# while True:

#     ret, frame = cap.read()

#     if not ret:
#         break

#     faces = face_cascade.detectMultiScale(
#         cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
#         1.3,
#         5
#     )

#     for (x,y,w,h) in faces:

#         face = frame[y:y+h, x:x+w]

#         face = cv2.resize(
#             face,
#             (IMG_SIZE,IMG_SIZE)
#         )

#         face = face.astype("float32") / 255.0

#         face = np.expand_dims(face, axis=0)

#         if reference_face is not None:

#             prediction = model.predict(
#                 [reference_face, face],
#                 verbose=0
#             )[0][0]

#             if prediction > 0.5:

#                 label = f"MATCH {prediction:.2f}"
#                 color = (0,255,0)

#             else:

#                 label = f"NOT MATCH {prediction:.2f}"
#                 color = (0,0,255)

#             cv2.putText(
#                 frame,
#                 label,
#                 (x,y-10),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.8,
#                 color,
#                 2
#             )

#         cv2.rectangle(
#             frame,
#             (x,y),
#             (x+w,y+h),
#             (255,0,0),
#             2
#         )

#     cv2.imshow(
#         "Face Recognition",
#         frame
#     )

#     key = cv2.waitKey(1)

#     if key == ord('s'):

#         if len(faces) > 0:

#             (x,y,w,h) = faces[0]

#             ref = frame[y:y+h, x:x+w]

#             ref = cv2.resize(
#                 ref,
#                 (IMG_SIZE,IMG_SIZE)
#             )

#             ref = ref.astype("float32") / 255.0

#             reference_face = np.expand_dims(
#                 ref,
#                 axis=0
#             )

#             print("Reference face captured")

#     if key == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()







# import cv2         aaj ka
# import keras
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import load_model

# # Allow loading models that contain Lambda layers
# keras.config.enable_unsafe_deserialization()

# IMG_SIZE = 100

# # Load trained model
# model = load_model("face_recognition_model.keras", custom_objects={"tf": tf})

# # Load face detector
# face_cascade = cv2.CascadeClassifier(
#     cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
# )

# cap = cv2.VideoCapture(0)
# print("Press 'S' to capture reference face")
# print("Press 'Q' to quit")

# reference_face = None

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, 1.3, 5)

#     for x, y, w, h in faces:
#         face = gray[y : y + h, x : x + w]
#         face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
#         face = face.astype("float32") / 255.0
#         face = np.reshape(face, (1, IMG_SIZE, IMG_SIZE, 1))

#         if reference_face is not None:
#             # Predict similarity between reference face and current face
#             prediction = model.predict([reference_face, face])[0][0]

#             if prediction > 0.5:
#                 label = "MATCH"
#                 color = (0, 255, 0)  # Green
#             else:
#                 label = "NOT MATCH"
#                 color = (0, 0, 255)  # Red

#             cv2.putText(
#                 frame,
#                 label,
#                 (x, y - 10),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.8,
#                 color,
#                 2,
#             )

#         # Draw bounding box for detected faces
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

#     cv2.imshow("Face Recognition", frame)
#     key = cv2.waitKey(1)

#     # Capture reference face
#     if key == ord("s") and len(faces) > 0:
#         x, y, w, h = faces[0]
#         ref = gray[y : y + h, x : x + w]
#         ref = cv2.resize(ref, (IMG_SIZE, IMG_SIZE))
#         ref = ref.astype("float32") / 255.0
#         reference_face = np.reshape(ref, (1, IMG_SIZE, IMG_SIZE, 1))
#         print("Reference face captured!")

#     # Quit application
#     if key == ord("q"):
#         break

# cap.release()
# cv2.destroyAllWindows()






# import cv2                                              pehle ka
# import numpy as np
# import keras
# import tensorflow as tf
# from tensorflow.keras.models import load_model

# # Allow loading models that contain Lambda layers
# keras.config.enable_unsafe_deserialization()

# IMG_SIZE = 100

# # Load trained model
# # model = load_model("face_recognition_model.keras")
# # model = load_model("face_recognition_model.keras", compile=False)
# model = load_model(
#     "face_recognition_model.keras",
#     custom_objects={"tf": tf}
# )

# # Load face detector
# face_cascade = cv2.CascadeClassifier(
#     cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
# )

# cap = cv2.VideoCapture(0)

# print("Press S to capture reference face")
# print("Press Q to quit")

# reference_face = None

# while True:

#     ret, frame = cap.read()
#     if not ret:
#         break

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     faces = face_cascade.detectMultiScale(gray, 1.3, 5)

#     for (x, y, w, h) in faces:

#         face = frame[y:y+h, x:x+w]
#         face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
#         face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
#         face = face.astype("float32") / 255.0
#         face = np.reshape(face, (1, IMG_SIZE, IMG_SIZE, 3))

#         if reference_face is not None:

#             prediction = model.predict([reference_face, face])[0][0]

#             if prediction > 0.5:
#                 label = "MATCH"
#                 color = (0, 255, 0)
#             else:
#                 label = "NOT MATCH"
#                 color = (0, 0, 255)

#             cv2.putText(
#                 frame,
#                 label,
#                 (x, y-10),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.8,
#                 color,
#                 2
#             )

#         cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

#     cv2.imshow("Face Recognition", frame)

#     key = cv2.waitKey(1)

#     # Capture reference face
#     if key == ord('s') and len(faces) > 0:

#         (x, y, w, h) = faces[0]

#         ref = frame[y:y+h, x:x+w]
#         ref = cv2.cvtColor(ref, cv2.COLOR_BGR2RGB)
#         ref = cv2.resize(ref, (IMG_SIZE, IMG_SIZE))
#         ref = ref.astype("float32") / 255.0

#         reference_face = np.reshape(ref, (1, IMG_SIZE, IMG_SIZE, 3))

#         print("Reference face captured")

#     if key == ord('q'):
#         break


# cap.release()
# cv2.destroyAllWindows()