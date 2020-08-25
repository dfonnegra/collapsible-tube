import os

import cv2 as cv
import joblib
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Conv2D,
    Conv2DTranspose,
    Dense,
    Flatten,
    Input,
    Reshape,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from img_preprocessing import mask_circle

THRESHOLD = 0.5
IMG_DIMENSION_W = 512  # No modificar, la red neuronal se entreno con estos valores
IMG_DIMENSION_H = 512  # No modificar, la red neuronal se entreno con estos valores
models = None
model_version = "2020-06-19 06:05:17.088259"
autoencoder_weights_path = f"../models/autoencoder_weights.h5"
classifier_local_path = f"../models/classifier.joblib"
scaler_local_path = f"../models/scaler.joblib"


class Hyperparameters:
    ALPHA = 2.3e-5
    IMG_DIMENSION_W = 512
    IMG_DIMENSION_H = 512
    FILTERS = (
        [32, 11, 4],
        [64, 7, 2],
        [64, 5, 2],
        [64, 3, 2],
        [128, 3, 2],
        [128, 3, 2],
        [256, 3, 2],
    )
    LATENT_DIM = 2056


def build_autoencoder():
    inputs = Input(
        shape=(Hyperparameters.IMG_DIMENSION_H, Hyperparameters.IMG_DIMENSION_W, 3)
    )
    x = inputs

    for filt in Hyperparameters.FILTERS:
        x = Conv2D(
            filt[0],
            (filt[1], filt[1]),
            strides=filt[2],
            padding="same",
            kernel_regularizer=keras.regularizers.l1_l2(l1=1e-4, l2=1e-4),
        )(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

    volume_size = K.int_shape(x)
    x = Flatten()(x)
    latent = Dense(
        Hyperparameters.LATENT_DIM,
        kernel_regularizer=keras.regularizers.l1_l2(l1=1e-4, l2=1e-4),
    )(x)
    encoder = Model(inputs, latent, name="encoder")

    latent_inputs = Input(shape=(Hyperparameters.LATENT_DIM,))
    x = Dense(
        np.prod(volume_size[1:]),
        kernel_regularizer=keras.regularizers.l1_l2(l1=1e-4, l2=1e-4),
    )(latent_inputs)
    x = Activation("relu")(x)
    x = Reshape((volume_size[1], volume_size[2], volume_size[3]))(x)

    for filt in Hyperparameters.FILTERS[::-1]:
        x = Conv2DTranspose(
            filt[0],
            (filt[1], filt[1]),
            strides=filt[2],
            padding="same",
            kernel_regularizer=keras.regularizers.l1_l2(l1=1e-4, l2=1e-4),
        )(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

    x = Conv2DTranspose(
        3,
        (5, 5),
        padding="same",
        kernel_regularizer=keras.regularizers.l1_l2(l1=1e-4, l2=1e-4),
    )(x)
    outputs = Activation("sigmoid")(x)
    decoder = Model(latent_inputs, outputs, name="decoder")
    autoencoder = Model(inputs, decoder(encoder(inputs)), name="autoencoder")
    autoencoder.compile(
        loss="binary_crossentropy",
        optimizer=Adam(learning_rate=Hyperparameters.ALPHA),
        #    optimizer=Adam(),
        metrics=["mse"],
    )
    return autoencoder


def get_dirt_predictor():
    global models
    if models is None:
        autoencoder = build_autoencoder()
        autoencoder.load_weights(autoencoder_weights_path)
        classifier = joblib.load(classifier_local_path)
        scaler = joblib.load(scaler_local_path)
        models = autoencoder, classifier, scaler
    return models


def format_image(img):
    return img.astype(np.float32) / 255.0


def predict_dirt(path_or_img):
    autoencoder, classifier, scaler = get_dirt_predictor()
    if type(path_or_img) == str:
        path_or_img = np.asarray(Image.open(path_or_img))
    img = cv.resize(cv.cvtColor(path_or_img, cv.COLOR_RGB2BGR), dsize=(640, 480))
    img = mask_circle(img)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img_real = format_image(img)
    img_pred = autoencoder.predict(np.array([img_real]))
    abs_diff = np.abs(img_real - img_pred)
    mse_vals = np.var(abs_diff)
    mean_vals = np.mean(abs_diff)
    count_vals = (abs_diff > 40 / 255).sum()
    X = np.array([[mse_vals, mean_vals, count_vals]])
    return classifier.predict_proba(scaler.transform(X))[0][1]
