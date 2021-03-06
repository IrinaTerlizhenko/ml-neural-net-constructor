import numpy as np

from netconstructor.layer import Layer


class ReluActivation(Layer):

    def __init__(self) -> None:
        super().__init__()

        self._current_input: np.ndarray = None

    def propagate(self, x: np.ndarray) -> np.ndarray:
        self._current_input = x
        return np.maximum(x, 0)

    def back_propagate(self, dx: np.ndarray) -> np.ndarray:
        relu_dx = self._current_input > 0  # ones when dx > 0 and zeros otherwise
        return relu_dx * dx  # component-wise


class LogisticActivation(Layer):
    THRESHOLD = -10

    def __init__(self) -> None:
        super().__init__()

        self._current_output: np.ndarray = None

    def propagate(self, x: np.ndarray) -> np.ndarray:
        threshold_x = np.clip(x, a_min=LogisticActivation.THRESHOLD, a_max=None)

        self._current_output = 1.0 / (1 + np.exp(-threshold_x))
        return self._current_output

    def back_propagate(self, dx: np.ndarray) -> np.ndarray:
        logistic_dx = self._current_output * (1 - self._current_output)  # component-wise
        return logistic_dx * dx  # component-wise


class LeakyReluActivation(Layer):

    def __init__(self, num_neurons: int, alpha: float = 0.01) -> None:
        if not 0 < alpha < 0.5:
            raise ValueError("alpha must be in the range (0, 0.5)")

        super().__init__()
        self._num_neurons = num_neurons
        self._alpha = alpha

        self._feed_func = np.vectorize(lambda x: x if x > 0. else self._alpha * x)
        self._back_func = np.vectorize(lambda x: 1. if x > 0. else self._alpha)

        self._current_output: np.ndarray = None

    def propagate(self, x: np.ndarray) -> np.ndarray:
        self._current_output = self._feed_func(x)
        return self._current_output

    def back_propagate(self, dx: np.ndarray) -> np.ndarray:
        relu_dx = self._back_func(self._current_output)
        return relu_dx * dx


class ParamReluActivation(Layer):

    def __init__(self, num_neurons: int, learning_rate: float, alpha: float = 0.01) -> None:
        if not 0 < alpha < 0.5:
            raise ValueError("alpha must be in the range (0, 0.5)")

        super().__init__()
        self._num_neurons = num_neurons
        self._learning_rate = learning_rate

        self._alpha = alpha

        self._feed_func = np.vectorize(lambda x: x if x > 0. else self._alpha * x)
        self._back_func = np.vectorize(lambda x: 1. if x > 0. else self._alpha)

        self._current_input: np.ndarray = None
        self._current_output: np.ndarray = None

    def propagate(self, x: np.ndarray) -> np.ndarray:
        self._current_input = x
        self._current_output = self._feed_func(x)
        return self._current_output

    def back_propagate(self, dx: np.ndarray) -> np.ndarray:
        dalpha = np.sum(np.clip(self._current_input, a_min=None, a_max=0.) * dx, axis=0)
        relu_dx = self._back_func(self._current_output)
        self._alpha -= self._learning_rate * dalpha  # TODO: SHOULD WE TAKE AVG OR UPDATE EVERY ELEMENT?
        return relu_dx * dx


class EluActivation(Layer):

    def __init__(self, alpha: float = 0.01) -> None:
        if alpha < 0:
            raise ValueError("alpha should be not less than 0")

        super().__init__()
        self._alpha = alpha

        self._current_output: np.ndarray = None

    def propagate(self, x: np.ndarray) -> np.ndarray:
        x[x <= 0] = self._alpha * (np.e ** x[x <= 0] - 1.)
        self._current_output = x
        return self._current_output

    def back_propagate(self, dx: np.ndarray) -> np.ndarray:
        relu_dx = self._current_output
        relu_dx[relu_dx > 0] = 1.
        relu_dx[relu_dx <= 0] = self._alpha * (np.e ** relu_dx[relu_dx <= 0.])
        return relu_dx * dx


class SoftmaxActivation(Layer):

    def __init__(self) -> None:
        super().__init__()

        self._current_output: np.ndarray = None

    def propagate(self, x: np.ndarray) -> np.ndarray:
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        self._current_output = exps / np.sum(exps, axis=1, keepdims=True)
        return self._current_output

    def back_propagate(self, dx: np.ndarray) -> np.ndarray:
        out_dx = []

        for batch_item, dx_batch_item in zip(self._current_output, dx):
            item_softmax_dx = np.diagflat(batch_item) - np.outer(batch_item, batch_item)
            out_dx.append(dx_batch_item @ item_softmax_dx)

        return np.array(out_dx)
