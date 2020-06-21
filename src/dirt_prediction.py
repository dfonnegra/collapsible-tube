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
        if not os.path.exists(dirt_predictor_local_path):
            dirt_predictor = tf.keras.models.load_model(dirt_predictor_path)
            dirt_predictor.save(dirt_predictor_local_path, save_format="tf")
        else:
            dirt_predictor = tf.keras.models.load_model(dirt_predictor_local_path)
    return dirt_predictor


def format_image(img):
    return (
        cv.resize(img, dsize=(IMG_DIMENSION_W, IMG_DIMENSION_H)).astype(np.float32)
        / 255.0
    )


def predict_dirt(path_or_img):
    model = get_dirt_predictor()
    if type(path_or_img) == str:
        path_or_img = np.asarray(Image.open(path_or_img))
    img = format_image(path_or_img)
    probabilities = model.predict(np.array([img]))
    predictions = probabilities > THRESHOLD
    return predictions[0]
