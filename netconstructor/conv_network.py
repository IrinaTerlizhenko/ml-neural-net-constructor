import logging
from typing import List, Tuple

import numpy as np

from netconstructor.activation import ReluActivation, LogisticActivation, EluActivation, SoftmaxActivation
from netconstructor.error import SquareError, Error, CrossEntropyError
from netconstructor.layer import Layer


class ConvolutionNeuralNetwork:

    def __init__(self, num_inputs: int, learning_rate: float = 0.2) -> None:
        self._num_inputs: int = num_inputs
        self._learning_rate = learning_rate

        self._layers: List[Layer] = []
        self._error: Error = None

    def train(self, x: np.ndarray, y: np.ndarray, num_iterations: int) -> float:
        if num_iterations <= 0:
            raise ValueError("Number of iterations must be a positive integer number")

        if len(x.shape) < 2:
            raise ValueError("The input must be at least a 2-dimensional array (for one grayscale image).")

        if len(x.shape) > 4:
            raise ValueError("The input must be at most a 4-dimensional array (for multiple RGB images).")

        # reshape to at least 3-dimensional array for batch processing
        output = x.copy() if len(x.shape) > 2 else x.copy().reshape((1,) + x.shape)
        # reshape to 4-dimensional array for multiple channels
        if len(output.shape) == 3:
            output = output.reshape(output.shape + (1,))

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
        pass

    def _reduce_error(self, output_errors: np.ndarray) -> float:
        return output_errors.sum()  # AXIS=1 if we want to see separate error for each batch element

    def with_square_error(self) -> "ConvolutionNeuralNetwork":
        self._error = SquareError()
        return self

    def with_cross_entropy_error(self) -> "ConvolutionNeuralNetwork":
        self._error = CrossEntropyError()
        return self

    def with_relu_activation(self) -> "ConvolutionNeuralNetwork":
        latest_layer, num_inputs = self._load_net_characteristics()

        new_layer = ReluActivation(num_inputs)

        self._layers.append(new_layer)
        return self

    def with_elu_activation(self) -> "ConvolutionNeuralNetwork":
        latest_layer, num_inputs = self._load_net_characteristics()

        new_layer = EluActivation(num_inputs)

        self._layers.append(new_layer)
        return self

    def with_logistic_activation(self) -> "ConvolutionNeuralNetwork":
        latest_layer, num_inputs = self._load_net_characteristics()

        new_layer = LogisticActivation(num_inputs)

        self._layers.append(new_layer)
        return self

    def with_softmax_activation(self) -> "ConvolutionNeuralNetwork":
        latest_layer, num_inputs = self._load_net_characteristics()

        new_layer = SoftmaxActivation(num_inputs)

        self._layers.append(new_layer)
        return self

    def _load_net_characteristics(self) -> Tuple[Layer, int]:
        latest_layer = self._layers[-1] if self._layers else None
        num_inputs = latest_layer.num_outputs if latest_layer else self._num_inputs
        return latest_layer, num_inputs
