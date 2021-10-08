import math
import numpy as np
import nnfs

nnfs.init()

layer_output = [4.8, 1.21, 2.385]

exp_values = np.exp(layer_output)

normalized_values = exp_values / np.sum(exp_values)

print(normalized_values)
print(sum(normalized_values))
