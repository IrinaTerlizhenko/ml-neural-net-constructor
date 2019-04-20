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
                        # convolved = batch_item * weights  # todo: np.tile(a,(3,1))
                        # # sum all to get a scalar
                        # convolved_scalar = np.sum(convolved)
                        # output[j][curr_x][curr_y][i] = convolved_scalar
                        sum_by_depth = np.sum(batch_item, axis=2)  # todo move to beginning
                        convolved = sum_by_depth * weights
                        # sum all to get a scalar
                        convolved_scalar = np.sum(convolved)
                        output[j][curr_x][curr_y][i] = convolved_scalar
                    curr_y += stride
                curr_x += stride

        return output

    def back_propagate(self, dx: np.ndarray) -> np.ndarray:
        pass

    @property
    def num_outputs(self):
        raise NotImplementedError()
