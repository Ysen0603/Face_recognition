import tensorflow as tf
from prepare_data import prepare_data
import json

def create_model(num_classes):
    base_model = tf.keras.applications.MobileNetV2(input_shape=(128, 128, 3), include_top=False, weights='imagenet')
    base_model.trainable = False

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    train_data, val_data, class_names = prepare_data()
    model = create_model(len(class_names))
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=3)
    ]
    
    history = model.fit(train_data, validation_data=val_data, epochs=10, callbacks=callbacks)
    
    model.save('./models/face_recognition_model.h5')
    
    with open('./models/class_names.json', 'w') as f:
        json.dump(class_names, f)