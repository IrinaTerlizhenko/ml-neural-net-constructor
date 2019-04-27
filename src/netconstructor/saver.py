import pickle

from netconstructor.network import NeuralNetwork


def save(network: NeuralNetwork, file):
    pickle.dump(network, file)


def load(file) -> NeuralNetwork:
    return pickle.load(file)


def save_as(network: NeuralNetwork, filename: str):
    with open(filename, 'wb') as file:
        save(network, file)


def load_as(filename: str) -> NeuralNetwork:
    with open(filename, 'rb') as file:
        return load(file)
