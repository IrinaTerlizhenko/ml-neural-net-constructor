from abc import ABC, abstractmethod

import numpy as np


class Error(ABC):

    @abstractmethod
    def propagate(self, y: np.ndarray, target: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def back_propagate(self, y: np.ndarray, target: np.ndarray) -> np.ndarray:
        pass


class SquareError(Error):

    def propagate(self, y: np.ndarray, target: np.ndarray) -> np.ndarray:
        return (target - y) ** 2 / 2.0

    def back_propagate(self, y: np.ndarray, target: np.ndarray) -> np.ndarray:
        return y - target
