import os

import cv2

from config.settings import DATASET_DIR
from utils.face_utils import detect_faces


def capture_faces(name, source=0, num_images=400):
    """Captures faces from a video source or camera and saves them to the dataset."""
    cap = cv2.VideoCapture(source)
    save_path = os.path.join(DATASET_DIR, name)
    os.makedirs(save_path, exist_ok=True)

    count = 0
    print(f"Starting capture for {name}. Press 'q' to stop.")

    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            break

        faces = detect_faces(frame)

        for x, y, w, h in faces:
            face = frame[y : y + h, x : x + w]
            face = cv2.resize(face, (128, 128))

            img_filename = os.path.join(save_path, f"{count}.jpg")
            cv2.imwrite(img_filename, face)
            count += 1

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"Captured: {count}/{num_images}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        cv2.imshow("Face Capture", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Successfully captured {count} images for {name}.")


if __name__ == "__main__":
    name = input("Enter the person's name: ")
    capture_faces(name)
