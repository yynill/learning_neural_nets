import numpy as np
import json
import os


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        # Gradients of param
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        # Grafient on inputs
        self.dinputs = np.dot(dvalues, self.weights.T)


class Optimzer_SGD:
    def __init__(self, learning_rate=1.0, decay=0, momentum=0):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    # Call once before any parameter updates ​def p​ re_update_params​(​self​):
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer):

        # If we use momentum
        if self.momentum:

            # If layer does not contain momentum arrays, create them
            # filled with zeros
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)

            weight_updates = \
                self.momentum * layer.weight_momentums - \
                self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates

            bias_updates = \
                self.momentum * layer.bias_momentums - \
                self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates

        # Without momentum
        else:
            weight_updates = -self.current_learning_rate * \
                layer.dweights
            bias_updates = -self.current_learning_rate * \
                layer.dbiases

        layer.weights += weight_updates
        layer.biases += bias_updates

    def post_update_params(self):
        self.iterations += 1


class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        # becuase we need to modify inputs er make a copy
        self.dinputs = dvalues.copy()
        # Zero gradient where input values were negative
        self.dinputs[self.inputs <= 0] = 0


class Activation_Softmax:
    def forward(self, inputs):
        self.inputs = inputs
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)

        # Enumerate outputs and gradients
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            # Flatten output array
            single_output = single_output.reshape(-1, 1)
            # Calculate Jacobian matrix of the output
            jacobian_matrix = np.diagflat(
                single_output) - np.dot(single_output, single_output.T)
            # Calculate sample-wise gradient and add it to the array of sample gradients
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)


class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss


class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_prediction, y_true):
        y_prediction_clipped = np.clip(y_prediction, 1e-7, 1-1e-7)
        correct_confidences = np.sum(y_prediction_clipped*y_true, axis=1)
        neg_log_liklyhoods = -np.log(correct_confidences)
        return neg_log_liklyhoods

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        # Number of labels in every sample
        labels = 10  # hardcoded 10 (0-9)
        y_true = np.eye(labels)[y_true]

        # Calculate gradient
        self.dinputs = -y_true / dvalues
        # Normalize gradient (learning rate - dependent on amount of samples)
        self.dinputs = self.dinputs / samples


# Softmax classifier - combined Softmax activation
# and cross-entropy loss for faster backward step


class Activation_Softmax_Loss_CategoricalCrossentropy():
    # Creates activation and loss function objects
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()

    # Forward pass
    def forward(self, inputs, y_true):
        # Output layer's activation function
        self.activation.forward(inputs)
        # Set the output
        self.output = self.activation.output
        # Calculate and return loss value
        return self.loss.calculate(self.output, y_true)

    # Backward pass

    def backward(self, dvalues, y_true):

        # Number of samples
        samples = len(dvalues)

        # If labels are one-hot encoded,
        # turn them into discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        # Copy so we can safely modify
        self.dinputs = dvalues.copy()
        # Calculate gradient
        self.dinputs[range(samples), y_true] -= 1
        # Normalize gradient
        self.dinputs = self.dinputs / samples


def load_trainingData():
    try:
        if os.path.isfile('training_data.json') and os.path.getsize('training_data.json') > 0:
            with open('training_data.json', 'r') as json_file:
                training_data = json.load(json_file)
        else:
            training_data = []

        # Shuffle the training data
        np.random.shuffle(training_data)
        return training_data
    except:
        print("training_data file not found!")


def sort_trainingData(training_data):
    sample_count = len(training_data)
    for s in range(sample_count):
        entry = training_data[s]

        input_id = entry['input_id']
        number_value = entry['number_value']
        tile_values = entry['tile_values']

        X.append(tile_values)
        Y.append(number_value)


def one_hot_encode(class_labels):
    # Scalar values --> one_hot_encode
    num_classes = 10  # hardcodeed for 10 numbers 0-9
    one_hot_encoded = np.eye(num_classes)[class_labels]
    return one_hot_encoded


# input data All samples of 784 inputs
X = []

#  Class labels for each sample
Y = []

if __name__ == '__main__':
    training_data = load_trainingData()
    sort_trainingData(training_data)
    Y = one_hot_encode(Y)

    X = np.array(X)

    # Split data into batches
    batch_size = 64
    num_samples = len(training_data)
    num_batches = num_samples // batch_size

    input_layer = Layer_Dense(784, 16)
    hidden_layer = Layer_Dense(16, 10)

    activation = Activation_ReLU()
    activation_outout = Activation_Softmax()

    loss_function = Activation_Softmax_Loss_CategoricalCrossentropy()
    optimizer = Optimzer_SGD(learning_rate=0.05, decay=4e-4, momentum=0.)

    for batch_index in range(num_batches):
        correct_predictions = 0

        start_index = batch_index * batch_size
        end_index = start_index + batch_size

        # Extract batch data
        X_batch = X[start_index:end_index]
        Y_batch = Y[start_index:end_index]

        # Forward pass
        input_layer.forward(X_batch)
        activation.forward(input_layer.output)
        hidden_layer.forward(activation.output)
        activation_outout.forward(hidden_layer.output)

        # Calculate loss
        batch_loss = loss_function.forward(activation_outout.output, Y_batch)

        # Calculate predictions
        predictions = np.argmax(activation_outout.output, axis=1)
        true_labels = np.argmax(Y_batch, axis=1)

        # Count correct predictions
        correct_predictions += np.sum(predictions == true_labels)

        # Calculate accuracy
        accuracy = correct_predictions / len(Y_batch)

        # Backward pass
        loss_function.backward(loss_function.output, Y_batch)
        hidden_layer.backward(loss_function.dinputs)
        activation.backward(hidden_layer.dinputs)
        input_layer.backward(activation.dinputs)

        # Update weights and biases
        optimizer.pre_update_params()
        optimizer.pre_update_params()
        optimizer.update_params(input_layer)
        optimizer.update_params(hidden_layer)
        optimizer.post_update_params()
        optimizer.post_update_params()

        # Print loss for monitoring
        print(
            f"Batch {batch_index+1}/{num_batches}, Loss: {batch_loss}, Accuracy: {accuracy}")


# improve learning
# save best biases and weights or dont quit running the program
# show probailityies of when I draw
