from blocks import Dense
import numpy as np
import copy
from sklearn.metrics import mean_squared_error as mse

class Network:
    
    _delta = 0.0001
    
    def __init__(self, layers=[]):
        self.layers = np.array(layers) # initilaize random weights to the layers
        self.network_weights = [
            layer.layer_weights for layer in self.layers
        ]
        self.network_biases = [
            layer.layer_biases for layer in self.layers
        ]
        
        self.history_loss = np.inf

    def forward(self, X, layers):
        
        outputs = X.copy()
        
        for layer in layers:
            outputs = layer.output(outputs)
        return outputs
    
    def update_weights_and_biases(self, X, y, learning_rate=0.01):
        ref_layers = deepcopy(self.layers)
        updated_layers = deepcopy(self.layers)
        
        ref_outputs = self.forward(X, ref_layers)
        ref_loss = mse(y, ref_outputs) 
        
        ## weight updates
        for layer_idx in range(len(ref_layers)):
            dynamic_layers = deepcopy(self.layers)
            for neuron_idx in range(len(dynamic_layers[layer_idx].layer_weights)):
                for weight_idx in range(len(dynamic_layers[layer_idx].layer_weights[neuron_idx])):
                    dynamic_layers[layer_idx].layer_weights[neuron_idx][weight_idx] += Network._delta
                    
                    ##### forward #####
                    outputs = self.forward(X, dynamic_layers)
#                     print(outputs)
                    temp_loss = mse(y, outputs)
                    
                    #### get change in loss w.r.t change in weight
                    d_L = temp_loss - ref_loss
                    d_W = Network._delta
                    
                    dL_dW = d_L/d_W
                    
                    #### update weights
                    updated_layers[layer_idx].layer_weights[neuron_idx][weight_idx] -= learning_rate*dL_dW
                    
        ## bias updates
        for layer_idx in range(len(ref_layers)):
            dynamic_layers = deepcopy(self.layers)
            for neuron_idx in range(len(dynamic_layers[layer_idx].layer_biases)):
                dynamic_layers[layer_idx].layer_biases[neuron_idx] += Network._delta

                ##### forward #####
                outputs = self.forward(X, dynamic_layers)
                temp_loss = mse(y, outputs)
#                 print(f"Loss after changing weight by delta: {temp_loss}")
#                 print(f"Loss after before changing weight by delta: {ref_loss}")

                #### get change in loss w.r.t change in bias
                d_L = temp_loss - ref_loss
                d_B = Network._delta

                dL_dB = d_L/d_B

                #### update weights
                updated_layers[layer_idx].layer_biases[neuron_idx] -= learning_rate*dL_dB
        new_loss = mse(y, self.forward(X, updated_layers))
        self.layers = deepcopy(updated_layers)
        return new_loss
    
    def fit(self, X, y, batch_size, epochs=400, learning_rate=0.01):
        init_lr = learning_rate
        losses = []
        num_batches = np.int32(np.ceil(np.float32(len(X)) / batch_size))
        print(num_batches)
        for epoch in tqdm(iterable=range(epochs), desc=f'Training for {epochs} epochs', total=len(range(epochs))): # tqdm_notebook
            epoch_loss = 0
            for batch_num in range(num_batches):
                if batch_num == (num_batches - 1):
                    batch_X = X[batch_num*batch_size:,:]
                    batch_y = y[batch_num*batch_size:,:]
                else:
                    batch_X = X[batch_num*batch_size:(batch_num+1)*batch_size,:]
                    batch_y = y[batch_num*batch_size:(batch_num+1)*batch_size,:]
                new_loss = self.update_weights_and_biases(batch_X, batch_y, learning_rate=init_lr)
                epoch_loss += new_loss
            losses.append(epoch_loss)
        return losses
        
    def predict(self, X):
        outputs = self.forward(X, layers=self.layers)
        return outputs