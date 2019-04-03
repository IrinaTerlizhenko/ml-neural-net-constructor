import numpy as np

from nn.network import NeuralNetwork

w1 = np.array([
    [.15, .20],
    [.25, .30]
])

b1 = np.array([.35, .35])

w2 = np.array([
    [.40, .45],
    [.50, .55]
])

b2 = np.array([.60, .60])

x = np.array([.05, .10])

y = np.array([.01, .99])

network: NeuralNetwork = NeuralNetwork(2) \
    .with_dense_layer(2, w1, b1) \
    .with_logistic_activation() \
    .with_dense_layer(2, w2, b2) \
    .with_logistic_activation() \
    .with_square_error()

network.train(x, y, 10000)
