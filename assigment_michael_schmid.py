#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 16:26:04 2020

@author: nunigan
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import dtd_loader_color_patches

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")
