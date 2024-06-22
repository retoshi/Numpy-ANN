import numpy as np
from sklearn.metrics import mean_squared_error as mse

##### Activation Functions #####
def sigmoid(x):
    """
        This activation function returns a value (probability) between 0 and 1
    """
    
    return 1/(1+np.exp(-x))

def tanh(x):
    """
        This activation function returns a value between -1 and 1
    """
    
    return np.tanh(x)

def relu(x):
    """
        This activation function returns a value which is either 0 or 1
    """
    
    return np.maximum(x, 0)

def linear(x):
    """
        This activation function returns a value which is linear
    """
    
    return x


##### Neuron #####
class Neuron:
    """
        Neural Network Neuron Implementation
    """
    
    def __init__(self, input_weights, bias, activation):
        """
            input_weights: np.array() -> an array of shape n by 1 (n x 1) where n represents the number of inputs or activations from the previous layer
            bias: np.float64() -> a scalar value of shape 1 by 1
            activation: str -> the activation function for the neuron
        """
        
        self.activation = activation
        self.weights = input_weights # np.array of randomly initialized weights
        self.bias = bias # randomly initialized bias

    def __str__(self):
        return "Neuron <activation: %s>" % self.activation

    def __repr__(self):
        return "Neuron <activation: %s>" % self.activation

    def output(self, X):
        """
            X: np.array() -> an array of input values of shape m by n (m x n) into the neuron where m represents the number training examples or batch size and             n represents the number of inputs
        """
        
        z = np.dot(X, self.weights) + self.bias
        
        if self.activation == 'sigmoid':
            output = sigmoid(z)
            return output
        
        elif self.activation == 'tanh':
            output = tanh(z)
            return output
        
        elif self.activation == 'relu':
            output = relu(z)
            return output
        
        elif self.activation == 'linear':
            output = linear(z)
            return output

##### Layer #####

class Layer:
    """
        Neural Network Layer Implementation.
    """
    
    def __init__(self, num_inputs, num_units, activation='sigmoid'):
        """
            num_inputs: int -> number of inputs into the network layer
            num_units: int -> number of neuron units in a specific layer of a neural network
            activation: str -> activation for that particular layer
            self.neurons: np.array -> initialized neurons for the network
            self.layer_weights: np.array -> l by k (l x k) where l is the number of inputs from the previous layer and k is the number of neurons in the layer
        """
        
        self.num_units = num_units
        self.activation = activation
        
        self.neurons = np.array([
            Neuron(activation=self.activation, 
                   input_weights=np.random.randn(num_inputs), 
                   bias=np.random.randn()) for unit in range(self.num_units)]) # assigning random weights and biases
        
        self.layer_weights = np.array([neuron.weights for neuron in self.neurons]).T
        self.layer_biases = np.array([neuron.bias for neuron in self.neurons])

    def output(self, X):
        """
            X: np.array() -> an array of dimension m by l (m x l) where m is the number of training examples and l is the number of inputs from the previous                layer
            
            returns outputs -> an array of dimension m by k (m x k) where m is the number of training examples and k is the number of neurons in the layer
        """
        
        z = np.dot(X, self.layer_weights) + self.layer_biases
        
        if self.activation == 'sigmoid':
            output = sigmoid(z)
            return output
        
        elif self.activation == 'tanh':
            output = tanh(z)
            return output
        
        elif self.activation == 'relu':
            output = relu(z)
            return output
        
        elif self.activation == 'linear':
            output = linear(z)
            return output
    
#     def update_weights_and_biases(self, weights, biases):
#         self.weights = weights.T
#         self.biases = biases


if __name__ == "__main__":
    pass