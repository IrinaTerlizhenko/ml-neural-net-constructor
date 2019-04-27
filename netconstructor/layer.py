import logging
from abc import ABC, abstractmethod
from typing import Tuple, Callable

import math
import numpy as np


class Layer(ABC):

    @abstractmethod
    def propagate(self, x: np.ndarray) -> np.ndarray:
        pass

    def test(self, x: np.ndarray):
        return self.propagate(x)

    @abstractmethod
    def back_propagate(self, dx: np.ndarray) -> np.ndarray:
        pass


class DenseLayer(Layer):

    def __init__(self, num_outputs: int, learning_rate: float, initial_weights: np.ndarray = None,
                 initial_biases: np.ndarray = None, weights_initializer: Callable = None,
                 bias_initializer: Callable = None) -> None:
        self._num_outputs = num_outputs
        self._learning_rate = learning_rate

        self._weight_initializer = weights_initializer
        self._bias_initializer = bias_initializer

        self._weight = initial_weights
        self._bias = initial_biases

        self._current_inputs: np.ndarray = None

    def propagate(self, x: np.ndarray) -> np.ndarray:
        self._current_inputs = x

        num_inputs = x.shape[1]
        num_outputs = self._num_outputs

        if self._weight is None:
            if self._weight_initializer:
                self._weight = np.fromfunction(self._weight_initializer, (num_inputs, num_outputs))
            else:
                self._weight = np.random.uniform(-1. / math.sqrt(num_inputs), 1. / math.sqrt(num_inputs),
                                                 num_inputs * num_outputs).reshape(num_inputs, num_outputs)
        if self._bias is None:
            if self._bias_initializer:
                self._bias = np.fromfunction(self._bias_initializer, (num_outputs,))
            else:
                self._bias = np.zeros(shape=(1, num_outputs))

        return x.dot(self._weight) + self._bias

    def back_propagate(self, dx: np.ndarray) -> np.ndarray:
        output_dx = dx.dot(self._weight.T)

        cumulative_diff_weights = self._current_inputs.T @ dx
        self._weight -= self._learning_rate * cumulative_diff_weights

        logging.debug(f"weights: {self._weight}")

        cumulative_biases = np.sum(dx, 0)  # gradient for bias is always 1
        self._bias -= self._learning_rate * cumulative_biases

        return output_dx


class BatchNorm(Layer):
    EPSILON = 0.001

    def __init__(self, learning_rate: float, gamma: float = 1,
                 beta: float = 0) -> None:

        self._learning_rate = learning_rate

        self._gamma = gamma
        self._beta = beta

        self._current_iteration = 1
        self._moving_beta = self._beta
        self._moving_gamma = self._gamma

        self._current_inputs: np.ndarray = None

    def propagate(self, x: np.ndarray, is_train: bool = True) -> np.ndarray:
        self._current_inputs = x

        mu = np.mean(x, axis=0)
        sigma2 = np.var(x, axis=0)
        # Normalized x value
        hath = (x - mu) * (sigma2 + BatchNorm.EPSILON) ** (-0.5)

        if is_train:
            return self._gamma * hath + self._beta
        else:
            return self._moving_gamma * hath + self._moving_beta

    def test(self, x: np.ndarray):
        return self.propagate(x, False)

    def back_propagate(self, dx: np.ndarray) -> np.ndarray:
        x = self._current_inputs
        batch_size = x.shape[0]
        mu = np.mean(x, axis=0)
        var = np.var(x, axis=0)
        dbeta = np.sum(dx, axis=0)
        dgamma = np.sum((x - mu) * (var + BatchNorm.EPSILON) ** (-0.5) * dx, axis=0)
        # TODO: CHECK CORRECTNESS
        output_dx = (1. / batch_size) * self._gamma * (var + BatchNorm.EPSILON) ** (-0.5) \
                    * (
                            batch_size * dx - np.sum(dx, axis=0)
                            - (x - mu)
                            * (var + BatchNorm.EPSILON) ** (-1.0)
                            * np.sum(dx * (x - mu), axis=0)
                    )

        self._beta -= self._learning_rate * dbeta
        self._gamma -= self._learning_rate * dgamma

        self._moving_beta = ((self._moving_beta * self._current_iteration) + self._beta) / (
                self._current_iteration + 1)
        self._moving_gamma = ((self._moving_gamma * self._current_iteration) + self._gamma) / (
                self._current_iteration + 1)
        self._current_iteration += 1

        return output_dx


class ReshapeLayer(Layer):

    def __init__(self, new_shape: Tuple) -> None:
        self._new_shape = new_shape
        self._current_shape = None

    def propagate(self, x: np.ndarray) -> np.ndarray:
        self._current_shape = x.shape
        return x.reshape((x.shape[0],) + self._new_shape)

    def back_propagate(self, dx: np.ndarray) -> np.ndarray:
        return dx.reshape(self._current_shape)
