import numpy as np
import tempfile

from integration.utils import w1, b1, w2, b2, x, y
from netconstructor.network import NeuralNetwork
from netconstructor.saver import save, load


def test_save_trained_network_v2():
    network = _build_article_network()
    network.train(x, y, 40)

    pkl_file = tempfile.TemporaryFile()

    test_data = np.array([.3, .6])

    pre_save_result = network.test(test_data)

    save(network, pkl_file)

    pkl_file.seek(0)

    loaded_network = load(pkl_file)

    post_load_result = loaded_network.test(test_data)

    assert np.allclose(pre_save_result, post_load_result)


def _build_article_network() -> NeuralNetwork:
    return NeuralNetwork() \
        .with_dense_layer(2, w1, b1) \
        .with_logistic_activation() \
        .with_dense_layer(2, w2, b2) \
        .with_logistic_activation() \
        .with_square_error()
