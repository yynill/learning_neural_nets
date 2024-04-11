import numpy as np


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        # n_inputs by n_neurons - array
        # fill with random values in (-1, 1)
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        # fill biases with 0
        self.bias = np.zeros((1, n_neurons))

    def forward(self, inputs):
        # output = (k->n∑ input.k * weight.k) + bias
        self.output = np.dot(inputs, self.weights) + self.bias


class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


class Activation_Softmax:
    def forward(self, inputs):
        # Exponentialization y = e^x && Normalization
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities


class Loss:  # aka cost funtion
    def calculate(self, output, y):  # y:= intended target value
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss


class Loss_categoricalCrossEntropy(Loss):
    def forward(self, y_prediction, y_true):  # y_prediction = of NN, y_true = target value
        samples = len(y_prediction)
        y_prediction_clipped = np.clip(y_prediction, 1e-7, 1-1e-7)

        # Scalar values representing the class labels for each sample -  [0,1,1]
        # vs
        # one-hot encoded array:   [[1,0,0]
        #                           [0,1,0]
        #                           [0,1,0] ]
        # (different way of telling the system whats true)
        if len(y_true.shape) == 1:
            correct_confidences = y_prediction_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_prediction_clipped*y_true, axis=1)

        neg_log_liklyhoods = -np.log(correct_confidences)
        return neg_log_liklyhoods


Y = np.array([[1, 0, 0],  # For the first sample, the target is 0
              [0, 1, 0],  # For the second sample, the target is 1
              [0, 1, 0]])  # For the third sample, the target is 1

# alternative Scalar values representing the class labels for each sample
# Y = np.array([0, 1, 1])


# input data
# 3 samples of 4 inputs (Batch)
X = np.array([[1.0, 0.1, 1.7, 1.8],   # First sample
              [2.0, 1.2, 3.5, 9.0],   # Second sample
              [0.5, 2.3, 0.8, 4.1]])  # Third sample

densel = Layer_Dense(4, 3)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax()

densel. forward(X)
activation1.forward(densel.output)
dense2. forward(activation1.output)
activation2.forward(dense2.output)

loss_function = Loss_categoricalCrossEntropy()
loss = loss_function.calculate(activation2.output, Y)


print(loss)
