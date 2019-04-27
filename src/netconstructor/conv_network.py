import logging

import numpy as np

from netconstructor.conv_layer import ConvolutionLayer
from netconstructor.network import NeuralNetwork


class ConvolutionNeuralNetwork(NeuralNetwork):

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

    def with_convolution_layer(self, filters: np.ndarray, padding: int = 0,
                               stride: int = 1) -> "ConvolutionNeuralNetwork":

        new_layer = ConvolutionLayer(padding, stride, self._learning_rate, filters)

        self._layers.append(new_layer)
        return self
