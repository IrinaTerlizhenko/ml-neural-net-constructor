import logging

import numpy as np

from netconstructor.network import NeuralNetwork

logging.basicConfig(level=logging.DEBUG)

EXPECTED_ONE_ITERATION_ERROR = 0.2983711087600027

w1 = np.array([
    [.15, .20],
    [.25, .30]
])

b1 = np.array([.35, .35])

w2 = np.array([
    [.40, .45],
    [.50, .55]
])

b2 = np.array([.60, .60])

x = np.array([[.05, .10], [.05, .10], [.05, .10]])

y = np.array([[.01, .99], [.01, .99], [.01, .99]])


def test_elu():
    """
    Constructs the network from the article https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/,
    runs one iteration and checks the error on it.
    """

    network = _build_elu_network()

    error = network.train(x, y, 5)

    eps = 1e-5
    assert EXPECTED_ONE_ITERATION_ERROR + eps > error, \
        "Error on 5 iterations must be less than in article network"


def test_elu_10000():
    """
    Constructs the network from the article https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/,
    runs one iteration and checks the error on it.
    """

    network = _build_elu_network()

    network.train(x, y, 10000)


def test_batch_norm():
    """
    Constructs the network from the article https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/,
    runs one iteration and checks the error on it.
    """

    network = _build_batch_norm_network()

    error = network.train(x, y, 5)

    eps = 1e-5
    assert EXPECTED_ONE_ITERATION_ERROR + eps > error, \
        "Error on 5 iterations must be less than in article network"


def test_batch_norm_10000():
    """
    Constructs the network from the article https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/,
    runs one iteration and checks the error on it.
    """

    network = _build_batch_norm_network()

    network.train(x, y, 10000)


def test_batch_norm_after_activation():
    """
    Constructs the network from the article https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/,
    runs one iteration and checks the error on it.
    """

    network = _build_batch_norm_after_activation_network()

    error = network.train(x, y, 5)

    eps = 1e-5
    assert EXPECTED_ONE_ITERATION_ERROR + eps > error, \
        "Error on 5 iterations must be less than in article network"


def test_batch_norm_after_activation_10000():
    """
    Constructs the network from the article https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/,
    runs one iteration and checks the error on it.
    """

    network = _build_batch_norm_after_activation_network()

    network.train(x, y, 10000)


def _build_elu_network() -> NeuralNetwork:
    return NeuralNetwork(2) \
        .with_dense_layer(2, w1, b1) \
        .with_elu_activation() \
        .with_dense_layer(2, w2, b2) \
        .with_elu_activation() \
        .with_square_error()


def _build_batch_norm_network() -> NeuralNetwork:
    return NeuralNetwork(2) \
        .with_dense_layer(2, w1, b1) \
        .with_batch_norm() \
        .with_logistic_activation() \
        .with_dense_layer(2, w2, b2) \
        .with_batch_norm() \
        .with_logistic_activation() \
        .with_square_error()


def _build_batch_norm_after_activation_network() -> NeuralNetwork:
    return NeuralNetwork(2) \
        .with_dense_layer(2, w1, b1) \
        .with_logistic_activation() \
        .with_batch_norm() \
        .with_dense_layer(2, w2, b2) \
        .with_logistic_activation() \
        .with_batch_norm() \
        .with_square_error()
