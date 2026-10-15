import cv2
import os

student_name = input("Enter Student Name: ")

save_dir = os.path.join(
    "database",
    student_name
)

os.makedirs(
    save_dir,
    exist_ok=True
)

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)

count = 0
MAX_IMAGES = 50

print("Press SPACE to capture")
print("Press Q to quit")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:

        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            (0, 255, 0),
            2
        )

    cv2.putText(
        frame,
        f"Captured: {count}/{MAX_IMAGES}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow(
        "Student Enrollment",
        frame
    )

    key = cv2.waitKey(1)

    if key == ord(" "):

        if len(faces) > 0:

            x, y, w, h = faces[0]

            face = frame[
                y:y+h,
                x:x+w
            ]

            filename = os.path.join(
                save_dir,
                f"{count}.jpg"
            )

            cv2.imwrite(
                filename,
                face
            )

            count += 1

            print(
                f"Saved {filename}"
            )

    if key == ord("q"):
        break

    if count >= MAX_IMAGES:
        break

cap.release()
cv2.destroyAllWindows()

print("\nEnrollment Completed")