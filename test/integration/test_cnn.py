import logging

import numpy as np

from conv_network import ConvolutionNeuralNetwork
from datareader import load_from_img
from test import TEST_ROOT_DIR

logging.basicConfig(level=logging.INFO)


def test_cnn():
    network = _build_network()

    x = load_from_img(f'{TEST_ROOT_DIR}/resource/img/5.png')
    y = np.array([
        [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., ],
    ])

    one_iteration_error = network.train(x, y, 5)

    assert one_iteration_error < 0.5


def _build_network() -> ConvolutionNeuralNetwork:
    filters = np.ones((1, 3, 3))

    return ConvolutionNeuralNetwork(0.01) \
        .with_convolution_layer(filters) \
        .with_relu_activation() \
        .with_reshape_layer((63250 * len(filters),)) \
        .with_dense_layer(10) \
        .with_softmax_activation() \
        .with_square_error()
