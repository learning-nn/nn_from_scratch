import numpy as np

# converting to a layer with 4 input and 3 neuron
inputs = [1.2, 2.1, 3.4, 1.2]
weights = [[4.1, -4.5, 3.1, 2.3],
           [-4.1, 4.5, 2.1, 2.3],
           [4.1, 4.5, 3.1, -2.3]]
biases = [1, 2, 3]

layer_outputs = np.dot(weights, inputs) + biases
print(layer_outputs)
