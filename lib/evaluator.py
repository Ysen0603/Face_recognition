import json

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

from config.settings import CLASS_NAMES_PATH, MODEL_PATH
from lib.data_loader import prepare_data
from lib.model import load_recognition_model


def evaluate():
    """Evaluates the model and displays a confusion matrix."""
    model = load_recognition_model(MODEL_PATH)
    _, val_data, _ = prepare_data()

    with open(CLASS_NAMES_PATH, "r") as f:
        class_names = json.load(f)

    print("Evaluating model...")
    loss, accuracy = model.evaluate(val_data)
    print(f"Model accuracy: {accuracy * 100:.2f}%")

    # Generate predictions for confusion matrix
    y_pred_probs = model.predict(val_data)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = val_data.classes

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.show()


if __name__ == "__main__":
    evaluate()
