import numpy as np

from layer import Layer


class ConvolutionLayer(Layer):

    def __init__(self, padding: int, stride: int, learning_rate: float,
                 initial_weights: np.ndarray) -> None:
        self._padding = padding
        self._stride = stride
        self._learning_rate = learning_rate

        # todo: only odd square kernel size
        # todo: only kernels of same size
        self._filters = initial_weights

        self._current_inputs: np.ndarray = None

    def propagate(self, data: np.ndarray) -> np.ndarray:
        self._current_inputs = data.copy()

        pad = self._padding
        stride = self._stride
        kernel_size = self._filters.shape[1]

        output_x_dim = (data.shape[1] + pad * 2 - (kernel_size - 1)) // stride
        output_y_dim = (data.shape[2] + pad * 2 - (kernel_size - 1)) // stride
        output = np.zeros(shape=(data.shape[0], output_x_dim, output_y_dim, len(self._filters)))

        # don't want to pad by batch and depth dimension
        pad_mask = ((0, 0), (pad, pad), (pad, pad), (0, 0),)
        padded_data = np.pad(data, pad_mask, mode="edge")

        for i, weights in enumerate(self._filters):
            curr_x = 0
            while curr_x + kernel_size <= padded_data.shape[1]:
                curr_y = 0
                while curr_y + kernel_size <= padded_data.shape[2]:
                    window = padded_data[:, curr_x: curr_x + kernel_size, curr_y: curr_y + kernel_size, :]
                    for j, batch_item in enumerate(window):  # [curr_x:curr_x + stride, curr_y:curr_y + stride, :]
                        sum_by_depth = np.sum(batch_item, axis=2)
                        convolved = sum_by_depth * weights
                        # sum all to get a scalar
                        convolved_scalar = np.sum(convolved)
                        output[j][curr_x][curr_y][i] = convolved_scalar
                    curr_y += stride
                curr_x += stride

        return output

    def back_propagate(self, dx: np.ndarray) -> np.ndarray:

        data = self._current_inputs
        pad = self._padding
        stride = self._stride
        kernel_size = self._filters.shape[1]

        # don't want to pad by batch and depth dimension
        pad_mask = ((0, 0), (pad, pad), (pad, pad), (0, 0),)
        padded_data = np.pad(data, pad_mask, mode="edge")

        padded_output_dx = np.zeros(padded_data.shape)
        output_dw = np.zeros(self._filters.shape)
        for i, weights in enumerate(self._filters):
            curr_x = 0
            output_i = 0
            while curr_x + kernel_size <= padded_data.shape[1]:
                curr_y = 0
                output_j = 0
                while curr_y + kernel_size <= padded_data.shape[2]:
                    window = padded_data[:, curr_x: curr_x + kernel_size, curr_y: curr_y + kernel_size, :]
                    summed_window = np.sum(window, axis=3)
                    for j, batch_item in enumerate(summed_window):  # [curr_x:curr_x + stride, curr_y:curr_y + stride]
                        output_dw[i] += dx[j][output_i][output_j][i] * batch_item
                        # [ batch : i : j : depth ]
                        padded_output_dx[j][curr_x: curr_x + kernel_size, curr_y: curr_y + kernel_size] += \
                            np.tile(dx[j][output_i][output_j][i] * weights, padded_output_dx.shape[3]).reshape(
                                (kernel_size, kernel_size, padded_output_dx.shape[3]))
                    curr_y += stride
                    output_j += 1
                curr_x += stride
                output_i += 1

        self._filters -= self._learning_rate * output_dw

        return padded_output_dx[:, pad:-pad, pad:-pad, :] if pad > 0 else padded_output_dx
