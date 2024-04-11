import numpy as np

X = [1.0, -2.0, 3.0]
W = [-3.0, -1.0, 2.0]
B = 1.0

# Forward pass
xw0 = X[0] * W[0]
xw1 = X[1] * W[1]
xw2 = X[2] * W[2]

# output
z = xw0 + xw1 + xw2 + B

# Relu activation
y = max(0, z)

# print(xw0, xw1, xw2, B)
# print(z)

# derivative value from next layer
dvalue = 1.0  # ouput schould be 1

# Derivative of ReLU and the chain rule
drelu_dz = dvalue * (1. if z > 0 else 0.)  # activation RELU
# print(drelu_dz)

# Partial derivatives of the multiplication, the chain rule
dsum_dxw0 = 1
dsum_dxw1 = 1
dsum_dxw2 = 1
dsum_db = 1
drelu_dxw0 = drelu_dz * dsum_dxw0
drelu_dxw1 = drelu_dz * dsum_dxw1
drelu_dxw2 = drelu_dz * dsum_dxw2
drelu_db = drelu_dz * dsum_db
# print(drelu_dxw0, drelu_dxw1, drelu_dxw2, drelu_db)

# Partial derivatives of the multiplication, the chain rule
dmul_dx0 = W[0]
dmul_dx1 = W[1]
dmul_dx2 = W[2]
dmul_dw0 = X[0]
dmul_dw1 = X[1]
dmul_dw2 = X[2]
drelu_dx0 = drelu_dxw0 * dmul_dx0
drelu_dw0 = drelu_dxw0 * dmul_dw0
drelu_dx1 = drelu_dxw1 * dmul_dx1
drelu_dw1 = drelu_dxw1 * dmul_dw1
drelu_dx2 = drelu_dxw2 * dmul_dx2
drelu_dw2 = drelu_dxw2 * dmul_dw2

# If a number representing one of these partial derivatives is bigger,
# it means that a small change in the corresponding input data or weight
# will have a larger impact on the output of the ReLU activation function during backpropagation
# print(drelu_dx0, drelu_dw0, drelu_dx1, drelu_dw1, drelu_dx2, drelu_dw2)


############################################################################
########################   multible nerrons   ##############################
############################################################################


# Passed in gradient from the next layer (of 3 samples)
dvalues = np.array([[1., 1., 1.],
                    [2., 2., 2.],
                    [3., 3., 3.]])

# We have 3 sets of inputs - samples
inputs = np.array([[1, 2, 3, 2.5],
                   [2., 5., -1., 2],
                   [-1.5, 2.7, 3.3, -0.8]])

# We have 3 sets of weights - one set for each neuron we have 4 inputs, thus 4 weights
# recall that we keep weights transposed
weights = np.array([[0.2, 0.8, -0.5, 1],
                    [0.5, -0.91, 0.26, -0.5],
                    [-0.26, -0.27, 0.17, 0.87]]).T

# One bias for each neuron
# biases are the row vector with a shape (1, neurons)
biases = np.array([[2, 3, 0.5]])

# Forward pass
layer_outputs = np.dot(inputs, weights) + biases  # Dense layer
relu_outputs = np.maximum(0, layer_outputs)  # ReLU activation

# Optimize and test backpropagation here
# ReLU activation - simulates derivative with respect to input values
# from next layer passed to current layer during backpropagation
drelu = relu_outputs.copy()
drelu[layer_outputs <= 0] = 0

# Dense layer - by what amplitude each node should get moved
dinputs = np.dot(drelu, weights.T)
dweights = np.dot(inputs.T, drelu)
dbiases = np.sum(drelu, axis=0, keepdims=True)

# update parameters
weights += -0.001 * dweights
biases += -0.001 * dbiases


print("dweights: ", dweights)
print("dbiases: ", dbiases)

print("New weights: ", weights)
print("New biases: ", biases)
