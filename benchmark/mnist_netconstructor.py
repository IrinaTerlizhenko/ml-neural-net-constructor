import time

import numpy as np
import tensorflow as tf

from netconstructor.network import NeuralNetwork

NUM_CLASSES = 10

mnist = tf.contrib.learn.datasets.load_dataset("mnist")

eye = np.eye(NUM_CLASSES)


def one_hot(labels):
    return np.array([eye[val] for val in labels], dtype=np.float)


# training
train_images = mnist.train.images
train_labels = one_hot(mnist.train.labels)

# validation
validation_images = mnist.validation.images
validation_labels = mnist.validation.labels

# test
test_images = mnist.test.images
test_labels = mnist.test.labels


def _build_softmax_network() -> NeuralNetwork:
    return NeuralNetwork(learning_rate=0.05) \
        .with_dense_layer(10, initial_weights=lambda i, j: 0., initial_biases=lambda i: 0.) \
        .with_softmax_activation() \
        .with_square_error()


def _build_logistic_network() -> NeuralNetwork:
    return NeuralNetwork(learning_rate=0.05) \
        .with_dense_layer(10, initial_weights=lambda i, j: 0., initial_biases=lambda i: 0.) \
        .with_logistic_activation() \
        .with_square_error()


num_features = len(train_images[0])
num_trainings = len(train_images)


def run_round(network, batch_size=30, num_epochs=30):
    for _ in range(num_epochs):
        total_batch = int(num_trainings / batch_size)

        for i in range(total_batch):
            batch_xs = train_images[i * batch_size:(i + 1) * batch_size]
            batch_ys = train_labels[i * batch_size:(i + 1) * batch_size]

            network.train(batch_xs, batch_ys, 1)


def validate(network):
    prediction = network.test(validation_images)
    label = np.argmax(prediction, axis=1)
    accuracy = np.sum(label == validation_labels) / len(validation_labels)

    print("Validation accuracy:", accuracy)
    return accuracy


def benchmark(network_creator):
    average = 0
    for _ in range(5):
        batch = 30
        epochs = 30

        network = network_creator()

        start = time.time()
        run_round(network, batch, epochs)
        end = time.time()
        print("_______________________________________________________")
        overall = end - start
        print(overall)
        print("_______________________________________________________")
        average += overall

        validate(network)

    average /= 5
    print("_______________________________________________________")
    print("Average:", average)
    print("_______________________________________________________")

    prediction = network.test(test_images)
    label = np.argmax(prediction, axis=1)
    accuracy = np.sum(label == test_labels) / len(test_labels)
    print("_______________________________________________________")
    print("Test accuracy:", accuracy)
    print("_______________________________________________________")


# benchmark(_build_softmax_network)
benchmark(_build_logistic_network)
