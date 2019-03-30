# todo: add parameter for degree
import numpy as np

from nn.layer import Layer


class ReluActivation(Layer):

    def __init__(self, num_neurons: int) -> None:
        super().__init__()
        self._num_neurons = num_neurons

        self._current_output: np.ndarray = None

    def propagate(self, x: np.ndarray) -> np.ndarray:
        return np.max(x, 0)

    def back_propagate(self, dx: np.ndarray) -> np.ndarray:
        relu_dx = self._current_output > 0  # ones when dx > 0 and zeros otherwise
        return relu_dx * dx  # component-wise

    @property
    def num_outputs(self):
        return self._num_neurons


class LogisticActivation(Layer):

    THRESHOLD = -10

    def __init__(self, num_neurons: int) -> None:
        super().__init__()
        self._num_neurons = num_neurons

        self._current_output: np.ndarray = None

    def propagate(self, x: np.ndarray) -> np.ndarray:
        threshold_x = np.clip(x, a_min=LogisticActivation.THRESHOLD, a_max=None)

        self._current_output = 1.0 / (1 + np.exp(-threshold_x))
        return self._current_output

    def back_propagate(self, dx: np.ndarray) -> np.ndarray:
        logistic_dx = self._current_output * (1 - self._current_output)  # component-wise
        return logistic_dx * dx  # component-wise

    @property
    def num_outputs(self):
        return self._num_neurons
