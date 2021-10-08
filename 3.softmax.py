import numpy as np
import nnfs

nnfs.init()

layer_output = [[4.8, 1.21, 2.385],
                [8.9, -1.81, 0.20],
                [1.41, 1.051, 0.026]]

exp_values = np.exp(layer_output)

normalized_values = exp_values / np.sum(exp_values, axis=1, keepdims=True)

print(normalized_values)
print(np.sum(normalized_values, axis=1, keepdims=True))
