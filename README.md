# ml-neural-net-constructor

## Usage

1. Install package from source code (not published to PyPi)

    ```bash
    $ git clone https://github.com/IrinaTerlizhenko/ml-neural-net-constructor.git
    $ cd ml-neural-net-constructor/src
    $ pip install .
    ```
    
1. Use it in your code

    ```python
    from netconstructor.network import NeuralNetwork
    net = NeuralNetwork()\
       .with_dense_layer(...)\
       .with_logistic_activation()\
       .with_square_error()
    net.train(x, y, num_iterations)
    error = net.test(x_test)
    ```

The sample can be found [here](dev_docs/netconstructor_sample.py). `numpy` and `tensorflow` are required in 
order to run this sample (`tensorflow` is used only to load MNIST data). They can be installed separately 
or with `pip install -r requirements/requirements_dev.txt` (note: this also installs a bunch of other dev 
dependencies).

## Benchmark results

Available on [this page](benchmark/Benchmark%20results.md).

## Development docs

1. [Technical task](dev_docs/Technical%20Task.md)
1. [Work done](dev_docs/Technical%20Task.md)

## Usage from source code

1. Install all required packages

    ```pip install -r requirements/requirements.txt```

## Development environment setup

1. Install all required packages

    ```pip install -r requirements/requirements_dev.txt```

1. Activate pytest if you're working in a JetBrains IDE

    *File > Settings > Tools > Python Integrated Tools > Testing > Default test runner*

