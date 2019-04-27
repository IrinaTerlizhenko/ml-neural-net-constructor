#Benchmarks

## MNIST

Network configuration:

1. Dense 784x10 layer
1. Softmax activation
1. Squared error

Batch size | Learning rate | num_epochs
-----------|---------------|-----------
30 | 0.05 | 30

Average on 5 iterations:

Library | Time | Test accuracy
--------|------|--------------
Tensorflow | 48.97s | 0.916
netconstructor | 58.04s | 0.9189
