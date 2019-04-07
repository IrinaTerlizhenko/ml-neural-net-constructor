from abc import ABC, abstractmethod

import numpy as np


class Error(ABC):

    @abstractmethod
    def propagate(self, output: np.ndarray, target: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def back_propagate(self, output: np.ndarray, target: np.ndarray) -> np.ndarray:
        pass


class SquareError(Error):

    def propagate(self, output: np.ndarray, target: np.ndarray) -> np.ndarray:
        return (target - output) ** 2 / 2.0

    def back_propagate(self, output: np.ndarray, target: np.ndarray) -> np.ndarray:
        return output - target


class CrossEntropyError(Error):

    EPSILON = 1e-12

    def propagate(self, output: np.ndarray, target: np.ndarray) -> np.ndarray:
        if (output < 0).any() or (output > 1).any():  # todo target ?
            raise ValueError("Output must be in the range (0, 1). Consider using softmax as the output layer.")
        # we allow zeros but they spoil computations
        output = np.clip(output, a_min=CrossEntropyError.EPSILON, a_max=None)

        log_likelihood = - target * np.log(output)  # + (1 - target) * (np.log(1 - output))
        return log_likelihood

    def back_propagate(self, output: np.ndarray, target: np.ndarray) -> np.ndarray:
        return - target / output  # + (1 - target) / (1 - output)
