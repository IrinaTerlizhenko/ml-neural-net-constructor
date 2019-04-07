import logging

import numpy as np
import tensorflow as tf

from netconstructor.network import NeuralNetwork

logging.basicConfig(level=logging.INFO)

NUM_CLASSES = 10

mnist = tf.contrib.learn.datasets.load_dataset("mnist")

eye = np.eye(NUM_CLASSES)


def one_hot(labels):
    return np.array([eye[val] for val in labels], dtype=np.float)


# training
train_images = mnist.train.images
train_labels = one_hot(mnist.train.labels)

# validation
validation_images = mnist.validation.images
validation_labels = one_hot(mnist.validation.labels)

# test
test_images = mnist.test.images
test_labels = one_hot(mnist.test.labels)


# todo
def test_mnist():
    network = _build_batch_norm_after_activation_network()

    batch_size = 1

    for i in range(0, 100):
        network.train(train_images[i * batch_size: (i + 1) * batch_size],
                      train_labels[i * batch_size: (i + 1) * batch_size], 1)
    # network.train(train_images[:1], train_labels[:1], 100)


def test_mnist_on_zeros():
    network = _build_batch_norm_after_activation_network()

    train = []
    labels = []
    for i in range(0, 10000):
        if train_labels[i][0] > 0.0:
            train.append(train_images[i])
            labels.append(train_labels[i])

    for train_image, train_label in zip(train, labels):
        network.train(train_image, train_label, 1)


def test_mnist_on_zeros_in_batch():
    network = _build_batch_norm_after_activation_network()

    batch_size = 2

    train = []
    labels = []
    for i in range(0, 10000):
        if train_labels[i][0] > 0.0:
            train.append(train_images[i])
            labels.append(train_labels[i])
    train = np.array(train)
    labels = np.array(labels)

    for i in range(0, int(len(train) / batch_size)):
        network.train(train[i*batch_size: (i+1)*batch_size], labels[i*batch_size: (i+1)*batch_size], batch_size)


def _build_batch_norm_after_activation_network() -> NeuralNetwork:
    return NeuralNetwork(784, learning_rate=0.1) \
        .with_dense_layer(10) \
        .with_softmax_activation() \
        .with_batch_norm() \
        .with_square_error()
