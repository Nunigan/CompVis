#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 16:26:04 2020

@author: nunigan
"""

import numpy as np
import tensorflow as tf

from tensorflow.keras.models import *
from tensorflow.keras.layers import *

import matplotlib.pyplot as plt
import glob
import imageio
import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# workaround for TF1.15 bug "Could not create cudnn handle: CUDNN_STATUS_INTERNAL_ERROR"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


num_classes = 47
batch_size = 16
epochs = 100


def one_class_per_image():


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
        test = imageio.imread(filename)
        y_train[i, imageio.imread(filename)[0][0]] = 1
    
    for i, filename in enumerate(sorted(glob.glob('dtd_val/label*.png'))):
        y_val[i, imageio.imread(filename)[0][0]] = 1
    
    for i, filename in enumerate(sorted(glob.glob('dtd_test/label*.png'))):
        y_test[i, imageio.imread(filename)[0][0]] = 1
    
    x_train = np.append(x_train, x_val, axis=0)
    
    y_train = np.append(y_train, y_val, axis=0)
    
    inputs = tf.keras.layers.Input(shape=(128,128,1))
    net = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu')(inputs)
    net = tf.keras.layers.MaxPooling2D((2,2))(net)
    net = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation='relu')(net)
    net = tf.keras.layers.MaxPooling2D((2,2))(net)
    net = tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), activation='relu')(net)
    net = tf.keras.layers.MaxPooling2D((2,2))(net)
    net = tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), activation='relu')(net)
    net = tf.keras.layers.MaxPooling2D((2,2))(net)
    net = tf.keras.layers.Conv2D(filters=1024, kernel_size=(3,3), activation='relu')(net)
    net = tf.keras.layers.MaxPooling2D((2,2))(net)
    
    net = tf.keras.layers.Flatten()(net)
    net = tf.keras.layers.Dense(512, activation='relu')(net)
    net = tf.keras.layers.Dropout(0.5)(net)
    net = tf.keras.layers.Dense(128, activation='relu')(net)
    net = tf.keras.layers.Dropout(0.5)(net)
    net = tf.keras.layers.Dense(num_classes, activation='softmax')(net)

    model = tf.keras.models.Model(inputs=inputs, outputs=net)

    print(model.summary())
    
    model.compile(optimizer = "adam", loss = 'categorical_crossentropy', metrics = ['accuracy'])
    
    model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=(x_test, y_test))
    
    score = model.evaluate(x_test, y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])


def semantic_seg_no_downsampling():

    x_train = np.zeros((1840,128,128,1), dtype=np.uint8)
    x_val = np.zeros((1840,128,128,1), dtype=np.uint8)
    x_test = np.zeros((1840,128,128,1), dtype=np.uint8)
    
    y_train = np.zeros((1840,128,128,47), dtype=np.uint8)
    y_val = np.zeros((1840,128,128,47), dtype=np.uint8)
    y_test = np.zeros((1840,128,128,47), dtype=np.uint8)
    
    
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
    
    x_train = np.append(x_train, x_val, axis=0)
    
    y_train = np.append(y_train, y_val, axis=0)
    
    
    inputs = tf.keras.layers.Input(shape=(128,128,1))
    net = tf.keras.layers.Conv2D(filters=16	, kernel_size=(3,3), padding = 'same', activation='relu')(inputs)
    net = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding = 'same', activation='relu')(net)
    net = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding = 'same', activation='relu')(net)
    net = tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding = 'same', activation='relu')(net)
    # net = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding = 'same', activation='relu')(net)
    # net = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding = 'same', activation='relu')(net)
    
    net = tf.keras.layers.Conv2D(filters=num_classes, kernel_size=1, activation=None, 
                                              kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.001, seed=None))(net)
    net = tf.keras.layers.Activation('softmax')(net)
    
    model = tf.keras.models.Model(inputs=inputs, outputs=net)

    
    print(model.summary())
    
    model.compile(optimizer = "adam", loss = 'categorical_crossentropy', metrics = ['accuracy'])
    
    model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=(x_test, y_test))
    
    score = model.evaluate(x_test, y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
    
    
def semantic_seg_downsampling():


    x_train = np.zeros((1840,128,128,1), dtype=np.uint8)
    x_val = np.zeros((1840,128,128,1), dtype=np.uint8)
    x_test = np.zeros((1840,128,128,1), dtype=np.uint8)
    
    y_train = np.zeros((1840,128,128,47), dtype=np.uint8)
    y_val = np.zeros((1840,128,128,47), dtype=np.uint8)
    y_test = np.zeros((1840,128,128	,47), dtype=np.uint8)
    
    
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
    
    x_train = np.append(x_train, x_val, axis=0)
    
    y_train = np.append(y_train, y_val, axis=0)


    inputs = tf.keras.layers.Input(shape=(128,128,1))
    net = tf.keras.layers.Conv2D(filters=16	, kernel_size=(3,3), padding = 'same', activation='relu')(inputs)
    net = tf.keras.layers.MaxPooling2D((2,2))(net)
    net = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding = 'same', activation='relu')(net)
    net = tf.keras.layers.MaxPooling2D((2,2))(net)
    net = tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding = 'same', activation='relu')(net)
    net = tf.keras.layers.MaxPooling2D((2,2))(net)
    net = tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding = 'same', activation='relu')(net)
    net = tf.keras.layers.MaxPooling2D((2,2))(net)
    net = tf.keras.layers.Conv2D(filters=1024, kernel_size=(3,3), padding = 'same', activation='relu')(net)
    
    net = tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=(3,3), strides=(2,2), padding = 'same', activation='relu')(net)
    net = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=(3,3), strides=(2,2), padding = 'same', activation='relu')(net)
    net = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=(3,3), strides=(2,2), padding = 'same', activation='relu')(net)
    net = tf.keras.layers.Conv2DTranspose(filters=num_classes, kernel_size=1, strides=(2,2), padding = 'same', activation='softmax')(net)
  
    model = tf.keras.models.Model(inputs=inputs, outputs=net)

    
    print(model.summary())
    
    model.compile(optimizer = "adam", loss = 'categorical_crossentropy', metrics = ['accuracy'])
    
    model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=(x_test, y_test))
    
    score = model.evaluate(x_test, y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])

def semantic_seg_unet():
    
    x_train = np.zeros((1840,128,128,1), dtype=np.uint8)
    x_val = np.zeros((1840,128,128,1), dtype=np.uint8)
    x_test = np.zeros((1840,128,128,1), dtype=np.uint8)
    
    y_train = np.zeros((1840,128,128,1), dtype=np.uint8)
    y_val = np.zeros((1840,128,128,1), dtype=np.uint8)
    y_test = np.zeros((1840,128,128	,1), dtype=np.uint8)
    
    
    for i, filename in enumerate(sorted(glob.glob('dtd_train/image_*.png'))):
        x_train[i] = imageio.imread(filename).reshape(128,128,1)
    
    for i, filename in enumerate(sorted(glob.glob('dtd_val/image_*.png'))):
        x_val[i] = imageio.imread(filename).reshape(128,128,1)
    
    for i, filename in enumerate(sorted(glob.glob('dtd_test/image_*.png'))):
        x_test[i] = imageio.imread(filename).reshape(128,128,1)
    
    for i, filename in enumerate(sorted(glob.glob('dtd_train/label*.png'))):
        # y_train[i, imageio.imread(filename)[0][0]] = 1
        y_train[i] = imageio.imread(filename).reshape(128,128,1)
    
    
    for i, filename in enumerate(sorted(glob.glob('dtd_val/label*.png'))):
        # y_val[i, imageio.imread(filename)[0][0]] = 1
        y_val[i] = imageio.imread(filename).reshape(128,128,1)
    
    for i, filename in enumerate(sorted(glob.glob('dtd_test/label*.png'))):
        # y_test[i, imageio.imread(filename)[0][0]] = 1
        y_test[i] = imageio.imread(filename).reshape(128,128,1)
    
    x_train = np.append(x_train, x_val, axis=0)
    
    y_train = np.append(y_train, y_val, axis=0)
    
    
    inputs = tf.keras.layers.Input(shape=(128,128,1))
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)
    
    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
    
    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
    
    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    
    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'softmax')(conv9)
    net = conv10

    model = tf.keras.models.Model(inputs=inputs, outputs=net)


    print(model.summary())
    
    model.compile(optimizer = "adam", loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
    
    model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=(x_test, y_test))
    
    score = model.evaluate(x_test, y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
    
    
if __name__ == "__main__":
    
    one_class_per_image()
    # semantic_seg_no_downsampling()
    # semantic_seg_downsampling()
    # semantic_seg_unet()

