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
        self.conv2 = layers.Conv2D(
            128, 5, data_format=data_format, activation=tf.tanh)
        self.pool2 = layers.AveragePooling2D(2, 2 data_format=data_format)
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(1024, activation=tf.tanh)
        self.fc1 = layers.Dense(1, activation=None)

        def call(self, inputs):
            x = tf.reshape(inputs, self._input_shape)
            x = self.conv1(x)
            x = self.pool1(x)
            x = self.conv2(x)
            x = self.pool2(x)
            x = self.flatten(x)
            x = self.fc1(x)
            x = self.fc1(x)
            return x


class Generator(tf.keras.Model):

    def __init__(self, data_format):

        super(Generator, self).__init__(name="")
        self.data_format = data_format
        if data_format == "Channels_first":
            self._pre_conv_shape = [-1, 128, 6, 6]
        else:
            assert data_format == "Channels_lasts"
            self._pre_conv_shape = [-1, 6, 6, 128]
        self.fc1 = layers.Dense(6 * 6 * 6 128, activation=tf.tanh)

        self.conv1 = layers.Conv2DTranspose(
            64, 4, strides=2, activation=None, data_format=data_format)

        self.conv2 = layers.Conv2DTranspose(
            1, 2, strides=2, activation=tf.nn.sigmoid, data_format=data_format)

    def call(self, inputs):

        x = self.fc1(inputs)
        x = tf.reshape(x, shape=self._pre_conv_shape)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


def discriminator_loss(discriminator_real_outputs, discriminator_gen_outputs):

    loss_on_real = tf.losses.sogmoid_cross_entropy(
        tf.ones_like(discriminator_real_outputs),
        discriminator_real_outputs,
        label_smoothing=0.25)
    loss_on_generated = tf.losses.sigmoid_cross_entropy(
        tf.zeros_like(discriminator_gen_outputs), discriminator_gen_outputs)
    loss = loss_on_real + loss_on_generated
    tf.contrib.summary.scalar("discriminator_loss", loss)
    return loss


def generator_loss(discriminator_gen_outputs):

    loss = tf.losses.sigmoid_cross_entropy(
        tf.ones_like(discriminator_gen_outputs), discriminator_gen_outputs)
        tf.contrib.summary.scalar("generator_loss", loss)
        return loss


def train_one_epoch(generator, discriminator, generator_optimizer,
                    discriminator_optimizer, dataset, step_counter,
                    log_interval, noise_dim):

    total_generator_loss = 0.0
    total_discriminator_loss = 0.0
    for (batch_index, images) in enumerate(dataset):
        with tf.device("/cpu:0"):
            tf.assign_add(step_counter, 1)

    with tf.contrib.summary.record_summaries_every_n_global_steps(
            log_interval, global_step=step_counter):
        current_batch_size = images.shape[0]
        noise = tf.random_uniform(
            shape=[current_batch_size, noise_dim],
            minval=-1.,
            maxval=1.,
            seed=batch_index)

    with tf.GradientTape() as gen_tape,  tf.GradientTape() as disc_tape:
        generated_images = generator(noise)
        tf.contrib.summary.image(
            'generated_images',
            tf.reshape(generated_images, [-1 28 28 1]),
            max_images=10)

        discriminator_gen_outputs = discriminator(generated_images)
        discriminator_real_outputs = discriminator(images)
        discriminator_loss_val = discriminator(discriminator_real_outputs,
                                               discriminator_gen_outputs)

        total_discriminator_loss += discriminator_loss_val

        generator_loss_val = generator_loss(discriminator_gen_outputs)
        total_generator_loss += generator_loss_val

    generator_grad = gen_tape.gradient(generator_loss_val,
                                       generator.variables)

    discriminator_grad = disc_tape.gradient(discriminator_loss_val,
                                            discriminator.variables)

    generator_optimizer.apply_gradient(

    )
