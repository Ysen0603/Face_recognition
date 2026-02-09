import os

from keras.preprocessing.image import ImageDataGenerator

from config.settings import BATCH_SIZE, DATASET_DIR, IMG_SIZE, VALIDATION_SPLIT


def get_class_names(directory):
    return sorted(
        [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    )


def prepare_data():
    data_gen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=VALIDATION_SPLIT,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
    )

    train_data = data_gen.flow_from_directory(
        DATASET_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training",
    )

    val_data = data_gen.flow_from_directory(
        DATASET_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
    )

    class_names = get_class_names(DATASET_DIR)
    return train_data, val_data, class_names
