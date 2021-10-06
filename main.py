import numpy as np

# converting to a layer with 4 input and 3 neuron
inputs = [[1.2, 2.1, 3.4, 1.2],
          [1.2, 2.1, 3.4, 1.2],
          [1.2, 2.1, 3.4, 1.2]]

weights = [[4.1, -4.5, 3.1, 2.3],
           [-4.1, 4.5, 2.1, 2.3],
           [4.1, 4.5, 3.1, -2.3]]
biases = [1, 2, 3]

weights2 = [[4.1, -4.5, 3.1],
           [-4.1, 4.5, 2.1],
           [4.1, 4.5, 3.1]]
biases2 = [1, 2, 3]

layer1_outputs = np.dot(inputs, np.array(weights).T) + biases
layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2
print(layer2_outputs)
