# converting to a layer with 4 input and 3 neuron
inputs = [1.2, 2.1, 3.4, 1.2]
weights = [[4.1, -4.5, 3.1, 2.3],
           [-4.1, 4.5, 2.1, 2.3],
           [4.1, 4.5, 3.1, -2.3]]
biases = [1, 2, 3]

layer_outputs = []
for neuron_weight, neuron_bias in zip(weights, biases):
    neuron_output = 0
    for n_input, weight in zip(inputs, neuron_weight):
        neuron_output += n_input*weight
    neuron_output += neuron_bias
    layer_outputs.append(neuron_output)

print(layer_outputs)
