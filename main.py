from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import argparse
import sys
import time
import os


FLAGS = None
layers = tf.keras.layers


class Discriminator(tf.keras.Model):
    """docstring for Discriminator."""

    def __init__(self, data_format):
        super(Discriminator, self).__init__()
        if data_format == "Channels_first":
            self._input_shape = [-1, 1, 28, 28]
        else:
            assert data_format == "Channels_lasts"
            self._input_shape = [-1, 28, 28, 1]
        self.conv1 = layers.Conv2D(
            64, 5, padding="SAME", data_format=data_format, activation=tf.tanh)
        self.pool1 = layers.AveragePooling2D(2, 2, data_format=data_format,
                                             activation=tf.tanh)
        self.conv2 = layers.Conv2D(
            128, 5, data_format=data_format, activation=tf.tanh)
        self.pool2 = layers.AveragePooling2D(2, 2, data_format=data_format)
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
        self.fc1 = layers.Dense(6 * 6 * 128, activation=tf.tanh)

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
            tf.reshape(generated_images, [-1, 28, 28, 1]),
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
        zip(generator_grad, generator.variables))
    discriminator.apply_gradient(
        zip(discriminator_grad, discriminator.variables))

    if log_interval and batch_index > 0 and batch_index % log_interval == 0:
        print("Batch #%d\tAverage Generator Loss: %.6f\t"
              "Average Discriminator Loss: %.6f" %
              (batch_index, total_generator_loss / batch_index,
               total_discriminator_loss / batch_index))


def main(_):
    (device, data_format) = ('/gpu:0', 'channels_first')
    if FLAGS.no_gpu or tf.contrib.eager.num_gpus() <= 0:
        (device, data_format) = ('/cpu:0', 'channels_last')
    print('Using device %s, and data format %s.' % (device, data_format))

    # Load the datasets
    data = input_data.read_data_sets(FLAGS.data_dir)
    dataset = (
        tf.data.Dataset.from_tensor_slices(data.train.images).shuffle(60000)
        .batch(FLAGS.batch_size))
    model_objects = {
        'generator': Generator(data_format),
        'discriminator': Discriminator(data_format),
        'generator_optimizer': tf.train.AdamOptimizer(FLAGS.lr),
        'discriminator_optimizer': tf.train.AdamOptimizer(FLAGS.lr),
        'step_counter': tf.train.get_or_create_global_step(),
    }

    summary_writer = tf.contrib.summary.create_summary_file_writer(
        FLAGS.output_dir, flush_millis=1000)
    checkpoint_prefix = os.path.join(FLAGS.checkpoint_dir, 'ckpt')
    latest_cpkt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    if latest_cpkt:
        print('Using latest checkpoint at ' + latest_cpkt)
    checkpoint = tf.train.Checkpoint(**model_objects)

    # Restore variables on creation if a checkpoint exists.
    checkpoint.restore(latest_cpkt)

    with tf.device(device):
        for _ in range(100):
            start = time.time()
            with summary_writer.as_default():
                train_one_epoch(dataset=dataset, log_interval=FLAGS.log_interval,
                                noise_dim=FLAGS.noise, **model_objects)

            end = time.time()
            checkpoint.save(checkpoint_prefix)
            print('\nTrain time for epoch #%d (step %d): %f' %
                  (checkpoint.save_counter.numpy(),
                   checkpoint.step_counter.numpy(),
                   end - start))


if __name__ == '__main__':
    tf.enable_eager_execution()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-dir',
        type=str,
        default='/tmp/tensorflow/mnist/input_data',
        help=('Directory for storing input data (default '
              '/tmp/tensorflow/mnist/input_data)'))
    parser.add_argument(
        '--batch-size',
        type=int,
        default=128,
        metavar='N',
        help='input batch size for training (default: 128)')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=100,
        metavar='N',
        help=('number of batches between logging and writing summaries '
              '(default: 100)'))
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        metavar='DIR',
        help='Directory to write TensorBoard summaries (defaults to none)')
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default='/tmp/tensorflow/mnist/checkpoints/',
        metavar='DIR',
        help=('Directory to save checkpoints in (once per epoch) (default '
              '/tmp/tensorflow/mnist/checkpoints/)'))
    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        metavar='LR',
        help='learning rate (default: 0.001)')
    parser.add_argument(
        '--noise',
        type=int,
        default=100,
        metavar='N',
        help='Length of noise vector for generator input (default: 100)')
    parser.add_argument(
        '--no-gpu',
        action='store_true',
        default=False,
        help='disables GPU usage even if a GPU is available')

    FLAGS, unparsed = parser.parse_known_args()
tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
