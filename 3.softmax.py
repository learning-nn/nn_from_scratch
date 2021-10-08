import math

layer_output = [4.8, 1.21, 2.385]

E = math.e

exp_values = []

for output in layer_output:
    exp_values.append(E**output)

normalized_values = []
normalized_base = sum(exp_values)

for value in exp_values:
    normalized_values.append(value / normalized_base)

print(normalized_values)
print(sum(normalized_values))
