#Benchmarks

## MNIST

Batch size | Learning rate | num_epochs
-----------|---------------|-----------
30 | 0.05 | 30

### Logistic activation

Network configuration:

1. Dense 784x10 layer
1. Logistic activation
1. Squared error

Average on 5 iterations:

Library | Time | Test accuracy
--------|------|--------------
Tensorflow | 50.16s | 0.9018
netconstructor | 60.46s | 0.9185

### Softmax activation

Network configuration:

1. Dense 784x10 layer
1. Softmax activation
1. Squared error

Average on 5 iterations:

Library | Time | Test accuracy
--------|------|--------------
Tensorflow | 48.97s | 0.916
netconstructor | 112.15s | 0.9229
