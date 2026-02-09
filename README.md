# Face Recognition System

A modular face recognition system built with TensorFlow, Keras, and OpenCV.

## Overview

This project provides a complete pipeline for face recognition, including face capture, data preparation, model training, evaluation, and prediction.

## Project Structure

- main.py: Entry point for the system via command-line interface.
- lib/: Core logic for data loading, model architecture, training, and prediction.
- config/: Configuration settings and constants.
- utils/: Common utility functions for face detection and preprocessing.
- models/: Directory to store trained models and class labels.
- dataset/: Directory where captured faces are stored (by name).
- haarcascades/: Haar Cascade XML files for initial face detection.

## Installation

This project uses `uv` for dependency management. To set up the environment:

```bash
uv sync
```

Dependencies:

- tensorflow
- opencv-python
- numpy
- scikit-learn
- seaborn
- matplotlib
- keras

## Usage

### 1. Capture Faces

To capture faces for a new person (using webcam):

```bash
python main.py capture --name "PersonName"
```

### 2. Train Model

To train the model on the captured dataset:

```bash
python main.py train
```

### 3. Evaluate Model

To evaluate the model's performance on the validation set:

```bash
python main.py evaluate
```

### 4. Predict

To predict faces in an image:

```bash
python main.py predict --image path/to/image.jpg
```

To predict faces in a video:

```bash
python main.py predict --video path/to/video.mp4
```

## Configuration

You can adjust hyperparameters and paths in `config/settings.py`, including:

- IMG_SIZE: Input dimensions for the neural network.
- BATCH_SIZE: Number of images per training batch.
- EPOCHS: Number of iterations for training.
- CONFIDENCE_THRESHOLD: Minimum confidence to consider a face "known".
