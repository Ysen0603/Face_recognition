import json

import cv2
import numpy as np

from config.settings import CLASS_NAMES_PATH, CONFIDENCE_THRESHOLD, MODEL_PATH
from lib.model import load_recognition_model
from utils.face_utils import detect_faces, preprocess_face


class FacePredictor:
    def __init__(self, model_path=MODEL_PATH, class_names_path=CLASS_NAMES_PATH):
        self.model = load_recognition_model(model_path)
        with open(class_names_path, "r") as f:
            self.class_names = json.load(f)

    def predict_frame(self, frame, confidence_threshold=CONFIDENCE_THRESHOLD):
        """Predicts faces in a single frame."""
        faces = detect_faces(frame)
        results = []

        for x, y, w, h in faces:
            face_img = frame[y : y + h, x : x + w]
            processed_face = preprocess_face(face_img)

            predictions = self.model.predict(processed_face, verbose=0)[0]
            max_confidence = np.max(predictions)
            label_index = np.argmax(predictions)

            if max_confidence < confidence_threshold:
                label = "unknown"
                confidence = 1 - max_confidence
            else:
                label = self.class_names[label_index]
                confidence = max_confidence

            results.append(
                {"label": label, "confidence": confidence, "box": (x, y, w, h)}
            )

        return results

    def annotate_frame(self, frame, results):
        """Annotates a frame with prediction results."""
        for res in results:
            x, y, w, h = res["box"]
            label = res["label"]
            confidence = res["confidence"]

            color = (0, 255, 0) if label != "unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                frame,
                f"{label} ({confidence:.2f})",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                color,
                2,
            )
        return frame


def predict_on_image(image_path):
    predictor = FacePredictor()
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return

    results = predictor.predict_frame(img)
    annotated_img = predictor.annotate_frame(img, results)

    cv2.imshow("Prediction", annotated_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return results


def predict_on_video(video_path):
    predictor = FacePredictor()
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = predictor.predict_frame(frame)
        annotated_frame = predictor.annotate_frame(frame, results)

        cv2.imshow("Video Prediction", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
