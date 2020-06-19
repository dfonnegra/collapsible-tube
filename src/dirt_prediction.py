import os

import cv2 as cv
import numpy as np
import tensorflow as tf
from PIL import Image

THRESHOLD = 0.5
IMG_DIMENSION_W = 640  # No modificar, la red neuronal se entreno con estos valores
IMG_DIMENSION_H = 480  # No modificar, la red neuronal se entreno con estos valores
dirt_predictor = None
model_version = "2020-06-19 06:05:17.088259"
dirt_predictor_path = f"gs://tubes_dataset_100/model/keras_export/{model_version}"
dirt_predictor_local_path = f"../models/{model_version}"


def get_dirt_predictor():
    global dirt_predictor
    if dirt_predictor is None:
        print("Here")
        if not os.path.exists(dirt_predictor_local_path):
            dirt_predictor = tf.keras.models.load_model(dirt_predictor_path)
            dirt_predictor.save(dirt_predictor_local_path, save_format="tf")
        else:
            dirt_predictor = tf.keras.models.load_model(dirt_predictor_local_path)
    return dirt_predictor


def read_image(file_path):
    img = np.asarray(Image.open(file_path))
    return (
        cv.resize(img, dsize=(IMG_DIMENSION_H, IMG_DIMENSION_W)).astype(np.float32)
        / 255.0
    )


def predict_dirt(path_or_img):
    model = get_dirt_predictor()
    if type(path_or_img) == str:
        image = read_image(path_or_img)
    elif type(path_or_img) == np.ndarray:
        image = format_image(path_or_img)
    else:
        raise ValueError(f"The type {type(path_or_img)} is not currently supported")
    probabilities = model.predict(np.array([image]))
    predictions = probabilities > THRESHOLD
    return predictions[0]
