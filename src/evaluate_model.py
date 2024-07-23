import tensorflow as tf
from prepare_data import prepare_data
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import json

def evaluate_model():
    model = tf.keras.models.load_model('./models/face_recognition_model.h5')
    _, val_data, _ = prepare_data()
    
    with open('./models/class_names.json', 'r') as f:
        class_names = json.load(f)
    
    loss, accuracy = model.evaluate(val_data)
    print(f"Model accuracy: {accuracy * 100:.2f}%")
    
    y_pred = model.predict(val_data)
    y_pred = np.argmax(y_pred, axis=1)
    y_true = val_data.classes
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

if __name__ == "__main__":
    evaluate_model()