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

def predict_on_video(video_path, confidence_threshold=0.8):
    model, class_names = load_model()
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file at {video_path}")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
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
            
            color = (0, 255, 0) if label != "unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f"{label} ({confidence:.2f})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = input("Enter the path of the video file: ")
    predict_on_video(video_path)
