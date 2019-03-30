from typing import List

import numpy as np

from nn.activation import ReluActivation, LogisticActivation
from nn.error import SquareError, Error
from nn.layer import Layer, DenseLayer


class Neuron:

    def __init__(self, bias: float, weights: List[float] = None) -> None:
        self._bias = bias
        # zero weight means no link
        self._weights = weights


class NeuralNetwork:

    # todo: different types of learning rate: constant, decreasing, etc.

    def __init__(self, num_inputs: int, learning_rate: float = 0.5) -> None:
        self._num_inputs: int = num_inputs
        self._learning_rate = learning_rate

        self._layers: List[Layer] = []
        self._error: Error = None

    def train(self, x: np.ndarray, y: np.ndarray, num_iterations: int) -> None:
        output = x.copy()

        for i in range(0, num_iterations):
            for layer in self._layers:
                output = layer.propagate(output)

            output_errors = self._error.propagate(output, y)
            error = self._reduce_error(output_errors)
            print(error)  # todo logging

            output = self._error.back_propagate(output, y)

            for layer in reversed(self._layers):
                output = layer.back_propagate(output)

    def _reduce_error(self, output_errors: np.ndarray) -> float:
        return output_errors.sum()

    def with_square_error(self) -> "NeuralNetwork":
        self._error = SquareError()
        return self

    def with_dense_layer(self, num_neurons: int, initial_weights: np.ndarray = None, initial_biases: np.ndarray = None
                         ) -> "NeuralNetwork":
        latest_layer = self._layers[-1] if self._layers else None
        num_inputs = latest_layer.num_outputs if latest_layer else self._num_inputs

        new_layer = DenseLayer(num_inputs, num_neurons, self._learning_rate, initial_weights, initial_biases)

        self._layers.append(new_layer)
        return self

    def with_relu_activation(self) -> "NeuralNetwork":
        latest_layer = self._layers[-1] if self._layers else None
        num_inputs = latest_layer.num_outputs if latest_layer else self._num_inputs

        new_layer = ReluActivation(num_inputs)

        self._layers.append(new_layer)
        return self

    def with_logistic_activation(self) -> "NeuralNetwork":
        latest_layer = self._layers[-1] if self._layers else None
        num_inputs = latest_layer.num_outputs if latest_layer else self._num_inputs

        new_layer = LogisticActivation(num_inputs)

        self._layers.append(new_layer)
        return self
