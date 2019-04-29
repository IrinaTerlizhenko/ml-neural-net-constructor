import logging
from typing import List, Tuple, Union, Callable

import numpy as np

from netconstructor.activation import ReluActivation, LogisticActivation, EluActivation, SoftmaxActivation
from netconstructor.error import SquareError, Error, CrossEntropyError
from netconstructor.layer import Layer, DenseLayer, BatchNorm, ReshapeLayer


class NeuralNetwork:

    def __init__(self, learning_rate: float = 0.2) -> None:
        self._learning_rate = learning_rate

        self._layers: List[Layer] = []
        self._error: Error = None

    def train(self, x: np.ndarray, y: np.ndarray, num_iterations: int) -> float:
        if num_iterations <= 0:
            raise ValueError("Number of iterations must be a positive integer number")

        if len(x.shape) > 2:
            raise ValueError("The input must be a 1- or 2-dimensional array.")

        # reshape to at least two dimensional array
        output = x.copy() if len(x.shape) > 1 else x.copy().reshape(1, len(x))
        y = y.copy()

        for i in range(0, num_iterations):
            for layer in self._layers:
                output = layer.propagate(output)

            output_errors = self._error.propagate(output, y)
            error = self._reduce_error(output_errors)
            logging.info(f"iteration: {i}, error: {error}")

            output = self._error.back_propagate(output, y)

            for layer in reversed(self._layers):
                output = layer.back_propagate(output)

        return error

    def test(self, x: np.ndarray) -> np.ndarray:
        if len(x.shape) > 2:
            raise ValueError("The input must be a 1- or 2-dimensional array.")

        # reshape to at least two dimensional array
        output = x.copy() if len(x.shape) > 1 else x.copy().reshape(1, len(x))

        for layer in self._layers:
            output = layer.test(output)

        return output

    def _reduce_error(self, output_errors: np.ndarray) -> float:
        return output_errors.sum()  # AXIS=1 if we want to see separate error for each batch element

    def with_square_error(self) -> "NeuralNetwork":
        self._error = SquareError()
        return self

    def with_cross_entropy_error(self) -> "NeuralNetwork":
        self._error = CrossEntropyError()
        return self

    def with_dense_layer(self, num_outputs: int, initial_weights: Union[np.ndarray, Callable] = None,
                         initial_biases: Union[np.ndarray, Callable] = None
                         ) -> "NeuralNetwork":

        weight = None
        weight_initializer = None
        if callable(initial_weights):
            weight_initializer = initial_weights
        elif type(initial_weights) is np.ndarray:
            weight = initial_weights

        bias = None
        bias_initializer = None
        if callable(initial_biases):
            bias_initializer = initial_biases
        elif type(initial_biases) is np.ndarray:
            bias = initial_biases

        new_layer = DenseLayer(num_outputs, self._learning_rate, initial_weights=weight, initial_biases=bias,
                               weights_initializer=weight_initializer, bias_initializer=bias_initializer)

        self._layers.append(new_layer)
        return self

    def with_batch_norm(self, gamma: float = 1, beta: float = 0) -> "NeuralNetwork":

        new_layer = BatchNorm(self._learning_rate, gamma, beta)

        self._layers.append(new_layer)
        return self

    def with_relu_activation(self) -> "NeuralNetwork":

        new_layer = ReluActivation()

        self._layers.append(new_layer)
        return self

    def with_elu_activation(self) -> "NeuralNetwork":

        new_layer = EluActivation()

        self._layers.append(new_layer)
        return self

    def with_logistic_activation(self) -> "NeuralNetwork":

        new_layer = LogisticActivation()

        self._layers.append(new_layer)
        return self

    def with_softmax_activation(self) -> "NeuralNetwork":

        new_layer = SoftmaxActivation()

        self._layers.append(new_layer)
        return self

    def with_reshape_layer(self, new_shape: Tuple) -> "NeuralNetwork":

        new_layer = ReshapeLayer(new_shape)

        self._layers.append(new_layer)
        return self
