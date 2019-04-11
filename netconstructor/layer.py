import logging
from abc import ABC, abstractmethod

import numpy as np
import math


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
            else np.random.uniform(-1. / math.sqrt(num_inputs), 1. / math.sqrt(num_inputs), num_inputs * num_outputs) \
            .reshape(num_inputs, num_outputs)
        self._bias = initial_biases if initial_biases is not None \
            else np.zeros(shape=(1, num_outputs))

        self._current_inputs: np.ndarray = None

    def propagate(self, x: np.ndarray) -> np.ndarray:
        self._current_inputs = x.copy()
        return x.dot(self._weight) + self._bias

    def back_propagate(self, dx: np.ndarray) -> np.ndarray:
        output_dx = dx.dot(self._weight.T)

        arr = []  # TODO: REFACTOR
        for single_input_obj, single_gradient in zip(self._current_inputs, dx):
            arr.append(np.outer(single_input_obj, single_gradient))
        diff_weights = np.array(arr)

        cumulative_diff_weights = np.sum(diff_weights, 0)
        self._weight -= self._learning_rate * cumulative_diff_weights

        logging.debug(f"weights: {self._weight}")

        cumulative_biases = np.sum(dx, 0)  # gradient for bias is always 1
        self._bias -= self._learning_rate * cumulative_biases

        return output_dx

    @property
    def num_outputs(self):
        return self._num_outputs


class BatchNorm(Layer):
    EPSILON = 0.001

    def __init__(self, num_outputs: int, learning_rate: float, gamma: float = 1,
                 beta: float = 0) -> None:  # TODO: IS_TRAIN (MOVING MEAN AND AVG)
        self._num_outputs = num_outputs

        self._learning_rate = learning_rate

        self._gamma = gamma
        self._beta = beta

        self._current_inputs: np.ndarray = None

    def propagate(self, x: np.ndarray) -> np.ndarray:
        self._current_inputs = x.copy()

        batch_size = x.shape[0]
        mu = 1 / batch_size * np.sum(x, axis=0)
        sigma2 = 1 / batch_size * np.sum((x - mu) ** 2, axis=0)
        hath = (x - mu) * (sigma2 + BatchNorm.EPSILON) ** (-1. / 2.)

        return self._gamma * hath + self._beta

    def back_propagate(self, dx: np.ndarray) -> np.ndarray:
        x = self._current_inputs
        batch_size = x.shape[0]
        mu = 1. / batch_size * np.sum(x, axis=0)
        var = 1. / batch_size * np.sum((x - mu) ** 2, axis=0)
        dbeta = np.sum(dx, axis=0)
        dgamma = np.sum((x - mu) * (var + BatchNorm.EPSILON) ** (-1. / 2.) * dx, axis=0)
        output_dx = (1. / batch_size) * self._gamma * (var + BatchNorm.EPSILON) ** (-1. / 2.) \
                    * (
                        batch_size * dx - np.sum(dx, axis=0)
                        - (x - mu)
                        * (var + BatchNorm.EPSILON) ** (-1.0)
                        * np.sum(dx * (x - mu), axis=0)
                    )

        self._beta -= self._learning_rate * dbeta  # TODO: SHOULD WE TAKE AVG OR UPDATE EVERY ELEMENT?
        self._gamma -= self._learning_rate * dgamma

        return output_dx

    @property
    def num_outputs(self):
        return self._num_outputs

class DropoutLayer(Layer):
    def __init__(self, num_inputs: int, dropout_rate: float) -> None:
        self._num_neurons = num_inputs
        self._dropout_rate = dropout_rate
        self.no_drop = False		# Set True to use all neurons on test & validation
        self._current_inputs: np.ndarray = None

    def propagate(self, x: np.ndarray) -> np.ndarray:
        self._current_inputs = x.copy()
    	if self.no_drop:
    		return x.dot(self._dropout_rate)
    	else:
	        self._dropout_matrix = np.diag(np.random.choice(2, size=self._num_neurons, p=[self._dropout_rate, 1 - self._dropout_rate]))
	        return x.dot(self._dropout_matrix)

    def back_propagate(self, dx: np.ndarray) -> np.ndarray:
        output_dx = dx.dot(self._dropout_matrix.T)

        return output_dx

    @property
    def num_outputs(self):
        return self._num_neurons