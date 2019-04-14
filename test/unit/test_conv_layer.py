import numpy as np

from conv_layer import ConvolutionLayer
from datareader import load_from_img
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
