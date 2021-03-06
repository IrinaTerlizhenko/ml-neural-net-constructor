import logging

import numpy as np

from netconstructor.network import NeuralNetwork
from integration.utils import w1, b1, w2, b2, x, y

X = np.array([1, 2, 3, 4])
X_BATCH = np.array([
    [1, 2, 3, 4],
    [5, 4, 3, 2],
])

Y = np.array([0.5, 1.0, 1.5])
Y_BATCH = np.array([
    [0.5, 1.0, 1.5],
    [2.0, 2.0, 2.0],
])

logging.basicConfig(level=logging.INFO)


def test_cross_entropy():
    net = _build_cross_entropy_network()

    net.train(X, Y, 1000)

    # todo: assert


def test_cross_entropy_batch():
    net = _build_cross_entropy_network()

    net.train(X_BATCH, Y_BATCH, 60)

    # todo: assert


def test_article_cross_entropy():
    net = _build_article_cross_entropy_network()

    net.train(x, y, 1000)

    # todo: assert


def _build_cross_entropy_network() -> NeuralNetwork:
    return NeuralNetwork() \
        .with_dense_layer(4) \
        .with_batch_norm() \
        .with_softmax_activation() \
        .with_dense_layer(3) \
        .with_batch_norm() \
        .with_softmax_activation() \
        .with_cross_entropy_error()


def _build_article_cross_entropy_network() -> NeuralNetwork:
    return NeuralNetwork(learning_rate=0.5) \
        .with_dense_layer(2, w1, b1) \
        .with_softmax_activation() \
        .with_dense_layer(2, w2, b2) \
        .with_softmax_activation() \
        .with_cross_entropy_error()
