import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

X, y = spiral_data(100, 3)


class LayerDense:

    def __init__(self, n_inputs, n_neurons):
        # changing n_inputs, n_neurons order so that we don't need to transpose weights
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        pass


class ActivationRelu:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


layer1 = LayerDense(2, 5)
activation1 = ActivationRelu()

layer1.forward(X)
activation1.forward(layer1.output)
print(activation1.output)

