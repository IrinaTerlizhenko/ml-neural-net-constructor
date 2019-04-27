import logging

import numpy as np

from netconstructor.network import NeuralNetwork

EXPECTED_ONE_ITERATION_ERROR = 0.2983711087600027

w1 = np.array([
    [.15, .20, .3],
    [.25, .30, .4]
])

b1 = np.array([.35, .35, .4])

x = np.array([[.05, .10], [.05, .10], [.05, .10], [.05, .10], ])

y = np.array([[.01, .99, 1.], [.01, .99, 1.], [.01, .99, 1.], [.01, .99, 1.], ])


logging.basicConfig(level=logging.INFO)


def test_network_article_multiple_iterations():
    network = _build_article_network()

    # back prop already influences the error
    error = network.train(x, y, 100)

    expected_error = 0.1
    assert error < expected_error


def _build_article_network() -> NeuralNetwork:
    return NeuralNetwork() \
        .with_dense_layer(3, w1, b1) \
        .with_logistic_activation() \
        .with_square_error()
