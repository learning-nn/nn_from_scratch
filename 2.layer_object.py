import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

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


class ActivationSoftmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities


class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss


class LossCategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods


X, y = spiral_data(100, 3)

dense1 = LayerDense(2, 3)
activation1 = ActivationRelu()
dense2 = LayerDense(3, 3)
activation2 = ActivationSoftmax()

dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

loss_function = LossCategoricalCrossEntropy()
loss = loss_function.calculate(activation2.output, y)

print(activation2.output[:5])
print('Loss:', loss)
