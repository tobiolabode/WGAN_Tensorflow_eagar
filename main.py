import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

layers = tf.keras.layers


class Discriminator(tf.keras.model):
    """docstring for Discriminator."""

    def __init__(self, data_format):
        super(Discriminator, self).__init__()
        if data_format == "Channels_first":
            self._input_shape = [-1, 1, 28, 28]
        else:
            assert data_format == "Channels_lasts"
            self._input_shape = [-1, 28, 28 1]
        self.conv1 = layers.Conv2D(
            64, 5, padding="SAME", data_format=data_format, activation=tf.tanh)
        self.pool1 = layers.AveragePooling2D(2, 2 data_format=data_format,
                                             activation=tf.tanh)
