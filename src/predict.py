import tensorflow as tf
import cv2
import numpy as np
import json

def load_model():
    model = tf.keras.models.load_model('./models/face_recognition_model.h5')
    with open('./models/class_names.json', 'r') as f:
        class_names = json.load(f)
    return model, class_names

def preprocess_image(image):
    image = cv2.resize(image, (128, 128))
    image = image / 255.0
    return np.expand_dims(image, axis=0)

def detect_face(image_path):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to read the image at {image_path}")
        return None, None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return img, faces

def predict(image_path, confidence_threshold=0.8):
    model, class_names = load_model()
    img, faces = detect_face(image_path)
    
    if img is None:
        return None
    
    if len(faces) == 0:
        print("No faces detected")
        return None

    results = []
    for (x, y, w, h) in faces:
        face = img[y:y+h, x:x+w]
        face = preprocess_image(face)
        predictions = model.predict(face)[0]
        max_confidence = np.max(predictions)
        label_index = np.argmax(predictions)
        
        if max_confidence < confidence_threshold:
            label = "unknown"
            confidence = 1 - max_confidence
        else:
            label = class_names[label_index]
            confidence = max_confidence
        
        results.append((label, confidence))
        
        color = (0, 255, 0) if label != "unknown" else (0, 0, 255)
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
        cv2.putText(img, f"{label} ({confidence:.2f})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    cv2.imshow("Prediction", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return results

if __name__ == "__main__":
    image_path = input("Enter the path of the image: ")
    results = predict(image_path)
    if results:
        for i, (label, confidence) in enumerate(results):
            print(f"Face {i+1}: Predicted: {label} with confidence {confidence:.2f}")
    else:
        print("No face detected or unable to predict.")