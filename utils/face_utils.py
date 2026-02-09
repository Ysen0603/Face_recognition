import cv2
import numpy as np

from config.settings import DEFAULT_HAARCASCADE, IMG_SIZE


def get_face_cascade():
    """Loads and returns the Haar Cascade classifier."""
    return cv2.CascadeClassifier(DEFAULT_HAARCASCADE)


def detect_faces(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)):
    """Detects faces in an image and returns coordinates."""
    face_cascade = get_face_cascade()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors, minSize=minSize
    )
    return faces


def preprocess_face(face_image):
    """Resizes and normalizes a face image for model prediction."""
    face = cv2.resize(face_image, IMG_SIZE)
    face = face / 255.0
    return np.expand_dims(face, axis=0)
