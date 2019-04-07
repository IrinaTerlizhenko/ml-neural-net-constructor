import logging

import numpy as np

from netconstructor.network import NeuralNetwork

logging.basicConfig(level=logging.INFO)

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
    network = _build_elu_network()

    error = network.train(x, y, 5)

    expected_error = 1e-3
    assert error < expected_error


def test_elu_multiple_iterations():
    network = _build_elu_network()

    error = network.train(x, y, 50)

    expected_error = 1e-30
    assert error < expected_error


def test_batch_norm():
    network = _build_batch_norm_network()

    error = network.train(x, y, 5)

    expected_error = 0.6
    assert error < expected_error


def test_batch_norm_multiple_iterations():
    network = _build_batch_norm_network()

    error = network.train(x, y, 10000)

    expected_error = 1e-4
    assert error < expected_error


def test_batch_norm_after_activation():
    network = _build_batch_norm_after_activation_network()

    error = network.train(x, y, 5)

    expected_error = 1e-2
    assert error < expected_error


def test_batch_norm_after_activation_multiple_iterations():
    network = _build_batch_norm_after_activation_network()

    error = network.train(x, y, 50)

    expected_error = 1e-30
    assert error < expected_error


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
