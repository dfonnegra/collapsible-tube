import os

import cv2 as cv
import joblib
import numpy as np
import tensorflow as tf
from PIL import Image

THRESHOLD = 0.5
IMG_DIMENSION_W = 512  # No modificar, la red neuronal se entreno con estos valores
IMG_DIMENSION_H = 512  # No modificar, la red neuronal se entreno con estos valores
models = None
model_version = "2020-06-19 06:05:17.088259"
autoencoder_local_path = f"../models/model_total_autoencoder.h5"
classifier_local_path = f"../models/classifier.joblib"
scaler_local_path = f"../models/scaler.joblib"


def get_dirt_predictor():
    global models
    if models is None:
        autoencoder = tf.keras.models.load_model(autoencoder_local_path)
        classifier = joblib.load(classifier_local_path)
        scaler = joblib.load(scaler_local_path)
        models = autoencoder, classifier, scaler
    return models


def format_image(img):
    return (
        cv.resize(img, dsize=(IMG_DIMENSION_W, IMG_DIMENSION_H)).astype(np.float32)
        / 255.0
    )


def predict_dirt(path_or_img):
    autoencoder, classifier, scaler = get_dirt_predictor()
    if type(path_or_img) == str:
        path_or_img = np.asarray(Image.open(path_or_img))
    img_real = format_image(path_or_img)
    img_pred = autoencoder.predict(np.array([img_real]))
    abs_diff = np.abs(img_real - img_pred)
    mse_vals = np.var(abs_diff)
    mean_vals = np.mean(abs_diff)
    count_vals = (abs_diff > 40 / 255).sum()
    X = np.array([[mse_vals, mean_vals, count_vals]])
    return classifier.predict(scaler.transform(X))[0]
