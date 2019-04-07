import logging

import numpy as np

from network import NeuralNetwork
from test.integration.utils import w1, b1, w2, b2, x, y


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


def test_softmax():
    net = _build_softmax_network()

    error = net.train(X, Y, 60)

    expected_error = 1e-10
    assert error < expected_error


def test_softmax_batch():
    net = _build_softmax_network()

    error = net.train(X_BATCH, Y_BATCH, 60)

    expected_error = 2.0
    assert error < expected_error


def test_article_softmax():
    net = _build_article_softmax_network()

    error = net.train(x, y, 60)

    expected_error = 1e-10
    assert error < expected_error


def _build_softmax_network() -> NeuralNetwork:
    return NeuralNetwork(4) \
        .with_dense_layer(4) \
        .with_softmax_activation() \
        .with_batch_norm() \
        .with_dense_layer(3) \
        .with_softmax_activation() \
        .with_batch_norm() \
        .with_square_error()


def _build_article_softmax_network() -> NeuralNetwork:
    return NeuralNetwork(2) \
        .with_dense_layer(2, w1, b1) \
        .with_softmax_activation() \
        .with_batch_norm() \
        .with_dense_layer(2, w2, b2) \
        .with_softmax_activation() \
        .with_batch_norm() \
        .with_square_error()
