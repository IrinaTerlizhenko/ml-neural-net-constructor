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


def test_network_article_forward_propagation():
    """
    Constructs the network from the article https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/,
    runs one iteration and checks the error on it.
    """

    network = _build_article_network()

    # actually no back prop when error is calculated
    network.train(x, y, 1)


def test_network_article_back_propagation_reduces_error():
    """
    Constructs the network from the article https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/,
    runs two iterations and checks the error on it.
    """

    network = _build_article_network()

    # back prop already influences the error
    network.train(x, y, 2)


def _build_article_network() -> NeuralNetwork:
    return NeuralNetwork(2) \
        .with_dense_layer(2, w1, b1) \
        .with_logistic_activation() \
        .with_square_error()
