#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 16:26:04 2020

@author: nunigan
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import glob
import imageio

import dtd_loader_color_patches

num_classes = 47
batch_size = 128
epochs = 20

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")

# %%

x_train = np.zeros((1840,128,128,1))
x_val = np.zeros((1840,128,128,1))
x_test = np.zeros((1840,128,128,1))

y_train = np.zeros((1840,47))
y_val = np.zeros((1840,47))
y_test = np.zeros((1840,47))


for i, filename in enumerate(sorted(glob.glob('dtd_train/image_*.png'))):
    x_train[i] = imageio.imread(filename).reshape(128,128,1)

for i, filename in enumerate(sorted(glob.glob('dtd_val/image_*.png'))):
    x_val[i] = imageio.imread(filename).reshape(128,128,1)

for i, filename in enumerate(sorted(glob.glob('dtd_test/image_*.png'))):
    x_test[i] = imageio.imread(filename).reshape(128,128,1)

for i, filename in enumerate(sorted(glob.glob('dtd_train/label*.png'))):
    y_train[i, imageio.imread(filename)[0][0]] = 1

for i, filename in enumerate(sorted(glob.glob('dtd_val/label*.png'))):
    y_val[i, imageio.imread(filename)[0][0]] = 1

for i, filename in enumerate(sorted(glob.glob('dtd_test/label*.png'))):
    y_test[i, imageio.imread(filename)[0][0]] = 1


# %% 

inputs = tf.keras.layers.Input(shape=(128,128,1))
net = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu')(inputs)
net = tf.keras.layers.MaxPooling2D((2,2))(net)
net = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu')(net)
net = tf.keras.layers.MaxPooling2D((2,2))(net)
net = tf.keras.layers.Flatten()(net)
net = tf.keras.layers.Dropout(0.5)(net)
net = tf.keras.layers.Dense(num_classes, activation='softmax')(net)


model = tf.keras.models.Model(inputs=inputs, outputs=net)

print(model.summary())

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])


model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(x_test, y_test),
                    callbacks = tensorboard_callback)

score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

