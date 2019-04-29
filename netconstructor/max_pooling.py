import numpy as np

from layer import Layer


class MaxpoolingLayer(Layer):

    def __init__(self, kernel_size: int, stride: int):
        self._current_inputs: np.ndarray = None
        self._max_indices: np.ndarray = None
        self._kernel_size = kernel_size
        self._stride = stride

    def propagate(self, data: np.ndarray) -> np.ndarray:
        self._current_inputs = data.copy()
        x_dim = data.shape[1]
        y_dim = data.shape[2]
        output_x_dim = 1 + (x_dim - 1) // self._stride
        output_y_dim = 1 + (y_dim - 1) // self._stride
        output = np.zeros(shape=(data.shape[0], output_x_dim, output_y_dim, data.shape[3]))
        self._max_indices = np.zeros(shape=output.shape, dtype=int)

        curr_x = 0
        while curr_x < x_dim:
            curr_y = 0
            while curr_y < y_dim:
                window = data[:, curr_x: min(curr_x + self._kernel_size, x_dim), curr_y: min(curr_y + self._kernel_size, y_dim), :]
                for c in range(data.shape[0]):
                    for f in range(data.shape[3]):
                        cur_window = window[c, :, :, f]
                        self._max_indices[c][curr_x // self._stride][curr_y // self._stride][f] = cur_window.argmax()
                        output[c][curr_x // self._stride][curr_y // self._stride][f] = np.max(cur_window)
                curr_y += self._stride
            curr_x += self._stride

        return output

    def back_propagate(self, dx: np.ndarray) -> np.ndarray:
        data = self._current_inputs
        output = np.zeros(shape = data.shape, dtype = int)

        for x in range(dx.shape[1]):
            for y in range(dx.shape[2]):
                for c in range(dx.shape[0]):
                    for f in range(dx.shape[3]):
                        max_ind = self._max_indices[c][x][y][f]
                        max_ind_unraveled = np.unravel_index(max_ind, [self._kernel_size,self._kernel_size])

                        if max_ind_unraveled[0] + x * self._stride >= output.shape[1]:
                            max_ind_unraveled[0] = 0
                        if max_ind_unraveled[1] + y * self._stride >= output.shape[2]:
                            max_ind_unraveled[1] = 0

                        output[c][max_ind_unraveled[0] + x * self._stride][max_ind_unraveled[1] + y * self._stride][f]\
                            = dx[c][x][y][f]
        return output
