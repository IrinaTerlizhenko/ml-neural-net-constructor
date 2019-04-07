import logging
from typing import List, Tuple

import numpy as np

from netconstructor.activation import ReluActivation, LogisticActivation, EluActivation, SoftmaxActivation
from netconstructor.error import SquareError, Error, CrossEntropyError
from netconstructor.layer import Layer, DenseLayer, BatchNorm


class NeuralNetwork:

    # todo: different types of learning rate: constant, decreasing, etc.

    def __init__(self, num_inputs: int, learning_rate: float = 0.2) -> None:
        self._num_inputs: int = num_inputs
        self._learning_rate = learning_rate

        self._layers: List[Layer] = []
        self._error: Error = None

    def train(self, x: np.ndarray, y: np.ndarray, num_iterations: int) -> float:
        if num_iterations <= 0:
            raise ValueError("Number of iterations must be a positive integer number")

        # reshape to at least two dimensional array
        output = x.copy() if len(x.shape) > 1 else x.copy().reshape(1, len(x))

        for i in range(0, num_iterations):
            for layer in self._layers:
                output = layer.propagate(output)

            output_errors = self._error.propagate(output, y)
            error = self._reduce_error(output_errors)
            logging.info(f"iteration: {i}, error: {error}")

            output = self._error.back_propagate(output, y)  # todo output_errors ?

            for layer in reversed(self._layers):
                output = layer.back_propagate(output)

        return error

    def _reduce_error(self, output_errors: np.ndarray) -> float:
        return output_errors.sum()  # TODO: AXIS=1

    def with_square_error(self) -> "NeuralNetwork":
        self._error = SquareError()
        return self

    def with_cross_entropy_error(self) -> "NeuralNetwork":
        self._error = CrossEntropyError()
        return self

    def with_dense_layer(self, num_outputs: int, initial_weights: np.ndarray = None, initial_biases: np.ndarray = None
                         ) -> "NeuralNetwork":
        latest_layer, num_inputs = self._load_net_characteristics()

        new_layer = DenseLayer(num_inputs, num_outputs, self._learning_rate, initial_weights, initial_biases)

        self._layers.append(new_layer)
        return self

    def with_batch_norm(self, gamma: float = 1, beta: float = 0) -> "NeuralNetwork":
        latest_layer, num_inputs = self._load_net_characteristics()

        new_layer = BatchNorm(num_inputs, self._learning_rate, gamma, beta)

        self._layers.append(new_layer)
        return self

    def with_relu_activation(self) -> "NeuralNetwork":
        latest_layer, num_inputs = self._load_net_characteristics()

        new_layer = ReluActivation(num_inputs)

        self._layers.append(new_layer)
        return self

    def with_elu_activation(self) -> "NeuralNetwork":
        latest_layer, num_inputs = self._load_net_characteristics()

        new_layer = EluActivation(num_inputs)

        self._layers.append(new_layer)
        return self

    def with_logistic_activation(self) -> "NeuralNetwork":
        latest_layer, num_inputs = self._load_net_characteristics()

        new_layer = LogisticActivation(num_inputs)

        self._layers.append(new_layer)
        return self

    def with_softmax_activation(self) -> "NeuralNetwork":
        latest_layer, num_inputs = self._load_net_characteristics()

        new_layer = SoftmaxActivation(num_inputs)

        self._layers.append(new_layer)
        return self

    def _load_net_characteristics(self) -> Tuple[Layer, int]:
        latest_layer = self._layers[-1] if self._layers else None
        num_inputs = latest_layer.num_outputs if latest_layer else self._num_inputs
        return latest_layer, num_inputs
