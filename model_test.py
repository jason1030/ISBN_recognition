from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib as mpl
import sys
import csv
from sklearn.model_selection import train_test_split
import cv2 as cv

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)

model = keras.models.load_model('my_ISBN_model')
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
img = [cv.resize(cv.imread('0-I.jpg', 0), (28, 28)).astype('float32')]
img = np.expand_dims(np.array(img) / 255, -1)

predictions = model.predict_classes(img).astype('int')
print((str(predictions[0])))
print(type(str(predictions[0])))
