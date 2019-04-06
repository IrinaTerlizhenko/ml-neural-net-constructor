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


def test_mnist_1000():
    network = _build_batch_norm_after_activation_network()

    for i in range(0, 100):
        network.train(train_images[i*10:(i+1)*10], train_labels[i*10:(i+1)*10], 1)


if __name__ == "__main__":
    test_mnist_1000()


def _build_batch_norm_after_activation_network() -> NeuralNetwork:
    return NeuralNetwork(784, learning_rate=0.1) \
        .with_dense_layer(10) \
        .with_relu_activation() \
        .with_batch_norm() \
        .with_square_error()
