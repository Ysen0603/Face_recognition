import os

# Base Directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
HAARCASCADES_DIR = os.path.join(BASE_DIR, "haarcascades")

# File Paths
MODEL_PATH = os.path.join(MODELS_DIR, "face_recognition_model.h5")
CLASS_NAMES_PATH = os.path.join(MODELS_DIR, "class_names.json")
DEFAULT_HAARCASCADE = os.path.join(HAARCASCADES_DIR, "haarcascade_frontalface_default.xml")

# Model Training Settings
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 10
VALIDATION_SPLIT = 0.2
CONFIDENCE_THRESHOLD = 0.8
