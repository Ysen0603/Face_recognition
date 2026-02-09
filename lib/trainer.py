import json

import tensorflow as tf

from config.settings import CLASS_NAMES_PATH, EPOCHS, MODEL_PATH
from lib.data_loader import prepare_data
from lib.model import create_model


def train():
    """Main training loop."""
    train_data, val_data, class_names = prepare_data()
    model = create_model(len(class_names))

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=3),
    ]

    print(f"Starting training for {len(class_names)} classes...")
    model.fit(train_data, validation_data=val_data, epochs=EPOCHS, callbacks=callbacks)

    # Save model and classes
    model.save(MODEL_PATH)
    with open(CLASS_NAMES_PATH, "w") as f:
        json.dump(class_names, f)

    print(f"Model saved to {MODEL_PATH}")
    print(f"Class names saved to {CLASS_NAMES_PATH}")


if __name__ == "__main__":
    train()
