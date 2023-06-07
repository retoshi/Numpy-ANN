import numpy as np
from sklearn.metrics import mean_squared_error as mse

##### Activation Functions #####
def sigmoid(x):
    return 1/1+np.exp(-x)

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(x, 0)

def linear(x):
    return x

##### Neuron #####

class Neuron:
    def __init__(self, weights, bias, activation='sigmoid'):
        self.activation = activation
        self.weights = weights # np.array of randomly initialized weights
        self.bias = bias # randomly initialized bias

    def __str__(self):
        return "Neuron <activation: %s>" % self.activation

    def __repr__(self):
        return "Neuron <activation: %s>" % self.activation

    def output(self, X):
        if self.activation == 'sigmoid':
            z = np.dot(X, self.weights) + self.bias
            g = sigmoid(z)
            return g
        elif self.activation == 'tanh':
            z = np.dot(X, self.weights) + self.bias
            g = tanh(z)
            return g
        elif self.activation == 'relu':
            z = np.dot(X, self.weights) + self.bias
            g = relu(z)
            return g
        elif self.activation == 'linear':
            z = np.dot(X, self.weights) + self.bias
            g = linear(z)
            return g

##### Dense Layer #####

class Dense:
     
     def __init__(self, units, input_shape, activation='sigmoid'):
        # assuming the input_shape to be a number for now
        self.units = units
        self.input_shape = input_shape
        self.neurons = [Neuron(activation=activation, weights=np.random.rand(input_shape), bias=np.random.rand()) for unit in range(units)] # assigning random weights and biases
        self.weights = np.array([neuron.weights for neuron in self.neurons]).T
        self.bias = np.array([neuron.bias for neuron in self.neurons])
        self.activation = activation
 
     def output(self, X):
        if self.activation == 'sigmoid':
            z = np.dot(X, self.weights) + self.bias
            g = sigmoid(z)
            return g
        elif self.activation == 'tanh':
            z = np.dot(X, self.weights) + self.bias
            g = tanh(z)
            return g
        elif self.activation == 'relu':
            z = np.dot(X, self.weights) + self.bias
            g = relu(z)
            return g
        elif self.activation == 'linear':
            z = np.dot(X, self.weights) + self.bias
            g = linear(z)
            return g
    
     def update_weights_and_biases(self, weights, biases):
        self.weights = weights.T
        self.biases = biases


if __name__ == "__main__":
    pass
    
