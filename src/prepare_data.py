import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import os

def get_class_names(directory):
    return sorted([d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))])

def prepare_data():
    data_gen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    train_data = data_gen.flow_from_directory(
        'dataset',
        target_size=(128, 128),
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )

    val_data = data_gen.flow_from_directory(
        'dataset',
        target_size=(128, 128),
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )

    class_names = get_class_names('dataset')
    return train_data, val_data, class_names

if __name__ == "__main__":
    train_data, val_data, class_names = prepare_data()
    print(f"Classes: {class_names}")