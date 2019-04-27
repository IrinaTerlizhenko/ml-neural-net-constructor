import numpy as np

from netconstructor.conv_layer import ConvolutionLayer
from netconstructor.datareader import load_from_img
from test import TEST_ROOT_DIR


def test_smoke():
    filters = np.array([
        [
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ],
    ])

    data = load_from_img(f'{TEST_ROOT_DIR}/resource/img/5.png')

    layer = ConvolutionLayer(1, 1, 1.0, filters)

    output = layer.propagate(data)

    summed_data = np.sum(data, axis=3)
    assert np.allclose(output, summed_data.reshape(summed_data.shape + (1,)))


def test_back_propagation():
    filters = np.array([
        [
            [0., 0., 0.],
            [0., 1., 0.],
            [0., 0., 0.],
        ],
    ])

    dx = np.array(
        [
            [1., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
        ],
    )
    dx = dx.reshape((1, 4, 4, 1))

    data = np.array(
        [
            [0., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
        ],
    )
    data = data.reshape((1, 4, 4, 1))

    layer = ConvolutionLayer(1, 1, 1.0, filters)

    layer.propagate(data)

    layer.back_propagate(dx)


def test_back_propagation_not_square_filter():
    filters = np.array([
        [
            [0., 0., 0.],
            [0., 1., 0.],
            [0., 0., 0.],
            [0., 0., 0.],
        ],
    ])

    dx = np.array(
        [
            [1., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
        ],
    )
    dx = dx.reshape((1, 3, 4, 1))

    data = np.array(
        [
            [0., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
        ],
    )
    data = data.reshape((1, 4, 4, 1))

    layer = ConvolutionLayer(1, 1, 1.0, filters)

    layer.propagate(data)

    layer.back_propagate(dx)
