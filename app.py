from flask import Flask, render_template, Response
import cv2
import sqlite3
import numpy as np
from collections import defaultdict
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

app = Flask(__name__)

# =====================================
# GLOBAL VARIABLES & CACHE
# =====================================
latest_name = "Waiting..."
latest_confidence = 0
latest_status = "No Detection"
attendance_cache = {}

# =====================================
# LOAD MODEL
# =====================================
embedding_model = load_model("embedding_model.keras", compile=False)
embeddings_db = np.load("embeddings.npy")
labels_db = np.load("labels.npy")

# =====================================
# CAMERA & DETECTOR CONFIG
# =====================================
camera = cv2.VideoCapture(0)
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

IMG_SIZE = 96
THRESHOLD =0 #0.72

# =====================================
# COSINE SIMILARITY
# =====================================
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# =====================================
# ATTENDANCE LOGGING
# =====================================
def mark_attendance(name):
    now = datetime.now()

    if name in attendance_cache:
        last_time = attendance_cache[name]
        seconds = (now - last_time).total_seconds()
        if seconds < 60:
            return

    attendance_cache[name] = now

    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()

    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")

    cursor.execute(
        """
        INSERT INTO attendance (name, date, time)
        VALUES (?, ?, ?)
        """,
        (name, date, time)
    )
    conn.commit()
    conn.close()

# =====================================
# FACE PREPROCESS
# =====================================
def preprocess_face(face):
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
    face = face.astype("float32")
    face = preprocess_input(face)
    return face

# =====================================
# VIDEO STREAM GENERATOR
# =====================================
def generate_frames():
    global latest_name, latest_confidence, latest_status
    
    while True:
        success, frame = camera.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.1, 5)

        for (x, y, w, h) in faces:
            face = frame[y : y + h, x : x + w]

            try:
                processed = preprocess_face(face)
                embedding = embedding_model.predict(
                    np.expand_dims(processed, axis=0), verbose=0
                )[0]
                embedding = embedding / np.linalg.norm(embedding)
                
                # =====================================
                # PERSON-WISE VOTING
                # =====================================
                person_scores = defaultdict(list)

                for db_embedding, db_label in zip(embeddings_db, labels_db):
                    score = cosine_similarity(embedding, db_embedding)
                    person_scores[db_label].append(score)

                best_name = "Unknown"
                best_score = -1

                for person, scores in person_scores.items():
                    # Take top 5 scores for that person
                    top_scores = sorted(scores, reverse=True)[:5]
                    avg_score = np.mean(top_scores)

                    print(f"{person} | Avg Score: {avg_score:.4f}")

                    if avg_score > best_score:
                        best_score = avg_score
                        best_name = person

                print(f"FINAL => {best_name} | {best_score:.4f}")

                if best_score > THRESHOLD:
                    latest_name = best_name
                    latest_confidence = round(best_score * 100, 2)
                    latest_status = "ENTRY ALLOWED"
                    color = (0, 255, 0)
                    mark_attendance(best_name)
                else:
                    latest_name = "Unknown"
                    latest_confidence = round(best_score * 100, 2)
                    latest_status = "ENTRY DENIED"
                    color = (0, 0, 255)

                    conn = sqlite3.connect("attendance.db")
                    cursor = conn.cursor()
                    now = datetime.now()
                    cursor.execute(
                        """
                        INSERT INTO unknown_log (date, time)
                        VALUES (?, ?)
                        """,
                        (now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")),
                    )
                    conn.commit()
                    conn.close()

                    best_name = "Unknown"

                # Draw bounding box and information text
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"{best_name} ({latest_confidence}%)", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            except Exception as e:
                pass

        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        )

# =====================================
# ROUTES
# =====================================
@app.route("/")
def home():
    return render_template(
        "index.html",
        latest_name=latest_name,
        latest_confidence=latest_confidence,
        latest_status=latest_status,
    )

@app.route("/video")
def video():
    return Response(
        generate_frames(), 
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

@app.route("/attendance")
def attendance():
    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM attendance")
    data = cursor.fetchall()
    conn.close()
    return render_template("attendance.html", data=data)

@app.route("/dashboard")
def dashboard():
    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM attendance")
    total_entries = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM unknown_log")
    unknown_count = cursor.fetchone()[0]

    cursor.execute("""
        SELECT name, COUNT(*)
        FROM attendance
        GROUP BY name
        ORDER BY COUNT(*) DESC
        LIMIT 5
    """)
    top_students = cursor.fetchall()

    cursor.execute("""
        SELECT date, COUNT(*)
        FROM attendance
        GROUP BY date
    """)
    chart_data = cursor.fetchall()

    dates = [x[0] for x in chart_data]
    counts = [x[1] for x in chart_data]

    current_time = datetime.now().strftime("%d %B %Y | %I:%M %p")
    conn.close()

    return render_template(
        "dashboard.html",
        total_entries=total_entries,
        unknown_count=unknown_count,
        current_time=current_time,
        top_students=top_students,
        dates=dates,
        counts=counts,
    )

if __name__ == "__main__":
    app.run(debug=True)





# from flask import Flask, render_template, Response
# import cv2
# import sqlite3
# import numpy as np
# from collections import defaultdict
# from datetime import datetime
# from tensorflow.keras.models import load_model
# from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# app = Flask(__name__)

# # =====================================
# # GLOBAL VARIABLES & CACHE
# # =====================================
# latest_name = "Waiting..."
# latest_confidence = 0
# latest_status = "No Detection"
# attendance_cache = {}

# # =====================================
# # LOAD MODEL
# # =====================================
# embedding_model = load_model("embedding_model.keras", compile=False)
# embeddings_db = np.load("embeddings.npy")
# labels_db = np.load("labels.npy")

# # =====================================
# # CAMERA & DETECTOR CONFIG
# # =====================================
# camera = cv2.VideoCapture(0)
# face_detector = cv2.CascadeClassifier(
#     cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
# )

# IMG_SIZE = 96
# # THRESHOLD = 0.85  below 3 lines are added 
# embedding_network.save(
#     "embedding_model.keras"
# )

# print(
#     "Embedding Model Saved Successfully!"
# )

# # =====================================
# # COSINE SIMILARITY
# # =====================================
# def cosine_similarity(a, b):
#     return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# # =====================================
# # ATTENDANCE LOGGING
# # =====================================
# def mark_attendance(name):
#     now = datetime.now()

#     if name in attendance_cache:
#         last_time = attendance_cache[name]
#         seconds = (now - last_time).total_seconds()
#         if seconds < 60:
#             return

#     attendance_cache[name] = now

#     conn = sqlite3.connect("attendance.db")
#     cursor = conn.cursor()

#     date = now.strftime("%Y-%m-%d")
#     time = now.strftime("%H:%M:%S")

#     cursor.execute(
#         """
#         INSERT INTO attendance (name, date, time)
#         VALUES (?, ?, ?)
#         """,
#         (name, date, time)
#     )
#     conn.commit()
#     conn.close()

# # =====================================
# # FACE PREPROCESS
# # =====================================
# def preprocess_face(face):
#     face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
#     face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
#     face = face.astype("float32")
#     face = preprocess_input(face)
#     return face

# # =====================================
# # VIDEO STREAM GENERATOR
# # =====================================
# def generate_frames():
#     global latest_name, latest_confidence, latest_status
    
#     while True:
#         success, frame = camera.read()
#         if not success:
#             break

#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = face_detector.detectMultiScale(gray, 1.1, 5)

#         for (x, y, w, h) in faces:
#             face = frame[y : y + h, x : x + w]

#             try:
#                 processed = preprocess_face(face)
#                 embedding = embedding_model.predict(
#                     np.expand_dims(processed, axis=0), verbose=0
#                 )[0]
#                 embedding = embedding / np.linalg.norm(embedding)

#                 # best_score = -1
#                 # best_name = "Unknown"

#                 # for db_embedding, db_label in zip(embeddings_db, labels_db):
#                 #      score = cosine_similarity(embedding, db_embedding)
#                 #      if score > best_score:
#                 #          best_score = score
#                 #          best_name = db_label
#                 # print(f"Matched: {db_label} | Score: {score:.4f}") added this line
                
#                 # =====================================
#                 # PERSON-WISE VOTING
#                 # =====================================
#                 person_scores = defaultdict(list)

#                 for db_embedding, db_label in zip(
#                         embeddings_db,
#                         labels_db):

#                     score = cosine_similarity(
#                         embedding,
#                         db_embedding
#                     )

#                     person_scores[db_label].append(score)

#                 best_name = "Unknown"
#                 best_score = -1

#                 for person, scores in person_scores.items():

#                   top_scores = sorted(
#                       scores,
#                       reverse=True
#                       )[:5]

#                   avg_score = np.mean(top_scores)

#                   max_score = max(top_scores)

#                   final_score = (
#                    0.7 * max_score
#                        +
#                    0.3 * avg_score
#                    )

#                   print(
#                    f"{person} | Score: {final_score:.4f}"
#                   )

#                   if final_score > best_score:
#                     best_score = final_score
#                     best_name = person

#                 print(
#                     f"FINAL => {best_name} | {best_score:.4f}"
#                 )

#                 if best_score > THRESHOLD:
#                     latest_name = best_name
#                     latest_confidence = round(best_score * 100, 2)
#                     latest_status = "ENTRY ALLOWED"
#                     color = (0, 255, 0)
#                     mark_attendance(best_name)
#                 else:
#                     latest_name = "Unknown"
#                     latest_confidence = round(best_score * 100, 2)
#                     latest_status = "ENTRY DENIED"
#                     color = (0, 0, 255)

#                     conn = sqlite3.connect("attendance.db")
#                     cursor = conn.cursor()
#                     now = datetime.now()
#                     cursor.execute(
#                         """
#                         INSERT INTO unknown_log (date, time)
#                         VALUES (?, ?)
#                         """,
#                         (now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")),
#                     )
#                     conn.commit()
#                     conn.close()

#                     best_name = "Unknown"

#                 # Optional: Draw a bounding box around the face on the frame
#                 cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
#                 cv2.putText(frame, f"{best_name} ({latest_confidence}%)", (x, y - 10),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

#             except Exception as e:
#                 pass

#         ret, buffer = cv2.imencode(".jpg", frame)
#         frame = buffer.tobytes()

#         yield (
#             b"--frame\r\n"
#             b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
#         )

# # =====================================
# # ROUTES
# # =====================================
# @app.route("/")
# def home():
#     return render_template(
#         "index.html",
#         latest_name=latest_name,
#         latest_confidence=latest_confidence,
#         latest_status=latest_status,
#     )

# @app.route("/video")
# def video():
#     return Response(
#         generate_frames(), 
#         mimetype="multipart/x-mixed-replace; boundary=frame"
#     )

# @app.route("/attendance")
# def attendance():
#     conn = sqlite3.connect("attendance.db")
#     cursor = conn.cursor()
#     cursor.execute("SELECT * FROM attendance")
#     data = cursor.fetchall()
#     conn.close()
#     return render_template("attendance.html", data=data)

# @app.route("/dashboard")
# def dashboard():
#     conn = sqlite3.connect("attendance.db")
#     cursor = conn.cursor()

#     cursor.execute("SELECT COUNT(*) FROM attendance")
#     total_entries = cursor.fetchone()[0]

#     cursor.execute("SELECT COUNT(*) FROM unknown_log")
#     unknown_count = cursor.fetchone()[0]

#     cursor.execute("""
#         SELECT name, COUNT(*)
#         FROM attendance
#         GROUP BY name
#         ORDER BY COUNT(*) DESC
#         LIMIT 5
#     """)
#     top_students = cursor.fetchall()

#     cursor.execute("""
#         SELECT date, COUNT(*)
#         FROM attendance
#         GROUP BY date
#     """)
#     chart_data = cursor.fetchall()

#     dates = [x[0] for x in chart_data]
#     counts = [x[1] for x in chart_data]

#     current_time = datetime.now().strftime("%d %B %Y | %I:%M %p")
#     conn.close()

#     return render_template(
#         "dashboard.html",
#         total_entries=total_entries,
#         unknown_count=unknown_count,
#         current_time=current_time,
#         top_students=top_students,
#         dates=dates,
#         counts=counts,
#     )

# if __name__ == "__main__":
#     app.run(debug=True)














# from flask import Flask, render_template, Response
# import cv2
# import sqlite3
# import numpy as np
# from collections import defaultdict
# from datetime import datetime
# from tensorflow.keras.models import load_model
# from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# app = Flask(__name__)

# # =====================================
# # GLOBAL VARIABLES & CACHE
# # =====================================
# latest_name = "Waiting..."
# latest_confidence = 0
# latest_status = "No Detection"
# attendance_cache = {}

# # =====================================
# # LOAD MODEL
# # =====================================
# embedding_model = load_model("embedding_model.keras", compile=False)
# embeddings_db = np.load("embeddings.npy")
# labels_db = np.load("labels.npy")

# # =====================================
# # CAMERA & DETECTOR CONFIG
# # =====================================
# camera = cv2.VideoCapture(0)
# face_detector = cv2.CascadeClassifier(
#     cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
# )

# IMG_SIZE = 96
# THRESHOLD =0.85

# # =====================================
# # COSINE SIMILARITY
# # =====================================
# def cosine_similarity(a, b):
#     return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# # =====================================
# # ATTENDANCE LOGGING
# # =====================================
# def mark_attendance(name):
#     now = datetime.now()

#     if name in attendance_cache:
#         last_time = attendance_cache[name]
#         seconds = (now - last_time).total_seconds()
#         if seconds < 60:
#             return

#     attendance_cache[name] = now

#     conn = sqlite3.connect("attendance.db")
#     cursor = conn.cursor()

#     date = now.strftime("%Y-%m-%d")
#     time = now.strftime("%H:%M:%S")

#     cursor.execute(
#         """
#         INSERT INTO attendance (name, date, time)
#         VALUES (?, ?, ?)
#         """,
#         (name, date, time)
#     )
#     conn.commit()
#     conn.close()

# # =====================================
# # FACE PREPROCESS
# # =====================================
# def preprocess_face(face):
#     face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
#     face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
#     face = face.astype("float32")
#     face = preprocess_input(face)
#     return face

# # =====================================
# # VIDEO STREAM GENERATOR
# # =====================================
# def generate_frames():
#     global latest_name, latest_confidence, latest_status
    
#     while True:
#         success, frame = camera.read()
#         if not success:
#             break

#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = face_detector.detectMultiScale(gray, 1.1, 5)

#         for (x, y, w, h) in faces:
#             face = frame[y : y + h, x : x + w]

#             try:
#                 processed = preprocess_face(face)
#                 embedding = embedding_model.predict(
#                     np.expand_dims(processed, axis=0), verbose=0
#                 )[0]
#                 embedding = embedding / np.linalg.norm(embedding)

#                 # best_score = -1
#                 # best_name = "Unknown"

#                 # for db_embedding, db_label in zip(embeddings_db, labels_db):
#                 #     score = cosine_similarity(embedding, db_embedding)
#                 #     if score > best_score:
#                 #         best_score = score
#                 #         best_name = db_label
#                 # print(f"Matched: {db_label} | Score: {score:.4f}") added this line
#                  # =====================================
# # PERSON-WISE VOTING
# # =====================================
# person_scores = defaultdict(list)

# for db_embedding, db_label in zip(
#         embeddings_db,
#         labels_db):

#     score = cosine_similarity(
#         embedding,
#         db_embedding
#     )

#     person_scores[db_label].append(score)

# best_name = "Unknown"
# best_score = -1

# for person, scores in person_scores.items():

#     # Take top 5 scores for that person
#     top_scores = sorted(
#         scores,
#         reverse=True
#     )[:5]

#     avg_score = np.mean(top_scores)

#     print(
#         f"{person} | Avg Score: {avg_score:.4f}"
#     )

#     if avg_score > best_score:
#         best_score = avg_score
#         best_name = person

# print(
#     f"FINAL => {best_name} | {best_score:.4f}"
# )

#                 if best_score > THRESHOLD:
#                     latest_name = best_name
#                     latest_confidence = round(best_score * 100, 2)
#                     latest_status = "ENTRY ALLOWED"
#                     color = (0, 255, 0)
#                     mark_attendance(best_name)
#                 else:
#                     latest_name = "Unknown"
#                     latest_confidence = round(best_score * 100, 2)
#                     latest_status = "ENTRY DENIED"
#                     color = (0, 0, 255)

#                     conn = sqlite3.connect("attendance.db")
#                     cursor = conn.cursor()
#                     now = datetime.now()
#                     cursor.execute(
#                         """
#                         INSERT INTO unknown_log (date, time)
#                         VALUES (?, ?)
#                         """,
#                         (now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")),
#                     )
#                     conn.commit()
#                     conn.close()

#                     best_name = "Unknown"

#                 # Optional: Draw a bounding box around the face on the frame
#                 cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
#                 cv2.putText(frame, f"{best_name} ({latest_confidence}%)", (x, y - 10),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

#             except Exception as e:
#                 pass

#         ret, buffer = cv2.imencode(".jpg", frame)
#         frame = buffer.tobytes()

#         yield (
#             b"--frame\r\n"
#             b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
#         )

# # =====================================
# # ROUTES
# # =====================================
# @app.route("/")
# def home():
#     return render_template(
#         "index.html",
#         latest_name=latest_name,
#         latest_confidence=latest_confidence,
#         latest_status=latest_status,
#     )

# @app.route("/video")
# def video():
#     return Response(
#         generate_frames(), 
#         mimetype="multipart/x-mixed-replace; boundary=frame"
#     )

# @app.route("/attendance")
# def attendance():
#     conn = sqlite3.connect("attendance.db")
#     cursor = conn.cursor()
#     cursor.execute("SELECT * FROM attendance")
#     data = cursor.fetchall()
#     conn.close()
#     return render_template("attendance.html", data=data)

# @app.route("/dashboard")
# def dashboard():
#     conn = sqlite3.connect("attendance.db")
#     cursor = conn.cursor()

#     cursor.execute("SELECT COUNT(*) FROM attendance")
#     total_entries = cursor.fetchone()[0]

#     cursor.execute("SELECT COUNT(*) FROM unknown_log")
#     unknown_count = cursor.fetchone()[0]

#     cursor.execute("""
#         SELECT name, COUNT(*)
#         FROM attendance
#         GROUP BY name
#         ORDER BY COUNT(*) DESC
#         LIMIT 5
#     """)
#     top_students = cursor.fetchall()

#     cursor.execute("""
#         SELECT date, COUNT(*)
#         FROM attendance
#         GROUP BY date
#     """)
#     chart_data = cursor.fetchall()

#     dates = [x[0] for x in chart_data]
#     counts = [x[1] for x in chart_data]

#     current_time = datetime.now().strftime("%d %B %Y | %I:%M %p")
#     conn.close()

#     return render_template(
#         "dashboard.html",
#         total_entries=total_entries,
#         unknown_count=unknown_count,
#         current_time=current_time,
#         top_students=top_students,
#         dates=dates,
#         counts=counts,
#     )

# if __name__ == "__main__":
#     app.run(debug=True)