# converting to a layer with 4 input and 3 neuron
inputs = [1.2, 2.1, 3.4, 1.2]
weight1 = [4.1, -4.5, 3.1, 2.3]
weight2 = [-4.1, 4.5, 2.1, 2.3]
weight3 = [4.1, 4.5, 3.1, -2.3]
bias1 = 1
bias2 = 2
bias3 = 3

output = [
    inputs[0] * weight1[0] + inputs[1] * weight1[1] + inputs[2] * weight1[2] + inputs[3] * weight1[3] + bias1,
    inputs[0] * weight2[0] + inputs[1] * weight2[1] + inputs[2] * weight2[2] + inputs[3] * weight2[3] + bias2,
    inputs[0] * weight3[0] + inputs[1] * weight3[1] + inputs[2] * weight3[2] + inputs[3] * weight3[3] + bias3,
    ]
print(output)
