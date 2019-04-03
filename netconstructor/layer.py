from abc import ABC, abstractmethod

import numpy as np


class Layer(ABC):

    @abstractmethod
    def propagate(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def back_propagate(self, dx: np.ndarray) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def num_outputs(self):
        pass


class DenseLayer(Layer):

    def __init__(self, num_inputs: int, num_outputs: int, learning_rate: float, initial_weights: np.ndarray = None,
                 initial_biases: np.ndarray = None) -> None:
        self._num_inputs = num_inputs
        self._num_outputs = num_outputs
        self._learning_rate = learning_rate

        self._weight = initial_weights if initial_weights is not None \
            else np.ones(shape=(num_inputs, num_outputs)) / 2.0
        self._bias = initial_biases if initial_biases is not None \
            else np.zeros(shape=(1, num_outputs))

        self._current_inputs: np.ndarray = None

    def propagate(self, x: np.ndarray) -> np.ndarray:
        self._current_inputs = x.copy()
        return self._weight.dot(x) + self._bias

    def back_propagate(self, dx: np.ndarray) -> np.ndarray:
        output_dx = self._weight.dot(dx)

        # diff_weights is something that we want to multiply by learning rate and subtract from each weight
        diff_weights = dx * self._current_inputs  # element-wise
        # but diff_weight contains unique values for each output of the previous layer
        # (i.e, is a vector of shape _num_inputs * 1)
        # we have _num_inputs * _num_outputs weights, but each bunch of size _num_outputs corresponds to the same output
        # (i.e. will have the same gradient)
        # so we need just to repeat diff_weights _num_outputs times and shape it correctly
        diff_weights = np.tile(diff_weights, self._num_outputs).reshape((self._num_outputs, self._num_inputs)).T
        self._weight -= self._learning_rate * diff_weights

        return output_dx

    @property
    def num_outputs(self):
        return self._num_outputs
