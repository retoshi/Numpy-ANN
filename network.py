from blocks import Dense
import numpy as np
import copy
from sklearn.metrics import mean_squared_error as mse


class Network:
    
    _delta = 0.01
    
    def __init__(self, input_shape, layers=[]):
        self.layers = layers # initilaize random weights to the layers
        self.weights = np.array([np.array([np.array([0.0 for feature in range(layer.input_shape)]) for unit in range(layer.units)]) for layer in self.layers])
        self.biases = np.array([np.array([0.0 for unit in range(layer.units)]) for layer in self.layers])
        
        #******************************************#
        self.assign_weights_to_layers()
        self.init_weights = copy.deepcopy(self.weights)
        self.init_biases = copy.deepcopy(self.biases)
        self.init_loss = 0
        self.loss_metric = ''
        
    def re_init(self,):
        self.weights = np.array([np.array([np.array([0.0 for feature in range(layer.input_shape)]) for unit in range(layer.units)]) for layer in self.layers])
        self.biases = np.array([np.array([0.0 for unit in range(layer.units)]) for layer in self.layers])
        
        #******************************************#
        self.assign_weights_to_layers()
        self.init_weights = copy.deepcopy(self.weights)
        self.init_biases = copy.deepcopy(self.biases)
        self.init_loss = 0
        self.loss_metric = ''
        
    def assign_weights_to_layers(self,):
        for layer_idx in range(len(self.weights)):
            for unit_idx in range(len(self.weights[layer_idx])):
                self.layers[layer_idx].neurons[unit_idx].weights = self.weights[layer_idx][unit_idx]
                self.layers[layer_idx].neurons[unit_idx].bias = self.biases[layer_idx][unit_idx]
        
    def weights_and_biases(self, ):
        return self.weights, self.biases
    
    def layers(self,):
        return self.layers
    
    def num_weights_and_biases(self, ):
        num_weights = 0
        num_biases = 0
        for layer in self.weights:
            for unit in layer:
                for weight in unit:
                    num_weights += 1
                    
        for layer in self.biases:
            for unit in layer:
                num_biases += 1
                
        return num_weights, num_biases
    
    def update_weights_and_biases(self, weights, biases):
        for layer_idx in range(len(weights)):
            self.layers[layer_idx].update_weights_and_biases(weights[layer_idx], biases[layer_idx])

    def output(self, X):
        outputs = X
        for layer in self.layers:
            outputs = layer.output(outputs)
        return outputs
    
    def loss_function(self, y_true, y_pred):
        if self.loss_metric == 'mse':
            return mse(y_true, y_pred)
        elif self.loss_metric == 'rmse':
            return mse(y_true, y_pred, squared=False)
        elif self.loss_metric == 'binary_crossentropy':
            return -1 * (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        elif self.loss_metric == 'sparse_categorical_crossentropy':
            pass
    
    def fit(self, X, y, loss_metric='mse', learning_rate=0.01, EPOCHS=100):
        self.re_init()
        self.loss_metric = loss_metric
        self.calculate_init_loss(X, y)
        
        # forward and backward propagations
        preds, loss = self.learn_params(X, y, learning_rate, EPOCHS)
        return preds, loss

    def calculate_init_loss(self, X, y):
        init_preds = self.output(X)
        self.init_loss = self.loss_function(init_preds, y)
    
    def reinitialize_weights_and_biases(self, weights, biases):
        self.init_weights = copy.deepcopy(weights)
        self.init_biases = copy.deepcopy(biases)
        
    def optimize_learning(self, X, y, loss_metric='mse'):
        best_eta = 0
        min_loss = np.inf
        possible_etas = np.arange(0,1,1e-4)
        
        for eta in possible_etas:
            preds, loss, = self.fit(X, y, loss_metric=loss_metric, learning_rate=eta, EPOCHS=100)
            if loss < min_loss:
                best_eta = eta
                min_loss = loss
                
        return self.fit(X, y, loss_metric=loss_metric, learning_rate=best_eta, EPOCHS=100), best_eta
    
    def predict(self, test):
        return self.output(test)
            
    def learn_params(self, X, y, learning_rate, EPOCHS):
        
        preds_history = []
        
        for i in range(EPOCHS):
            for layer_idx in range(len(self.weights)):
                for unit_idx in range(len(self.weights[layer_idx])):
                    for weight_idx in range(len(self.weights[layer_idx][unit_idx])):
                        weights_copy = copy.deepcopy(self.init_weights)
                        weights_copy[layer_idx][unit_idx][weight_idx] += 0.01

                        # get prediction from changing the current weight 
                        self.update_weights_and_biases(weights_copy, self.init_biases)
                        cur_pred = self.output(X)
                        
                        loss = self.loss_function(y, cur_pred)
                        dJ_dw = (loss - self.init_loss) / 0.01
                        self.weights[layer_idx][unit_idx][weight_idx] =- (learning_rate * dJ_dw)
                        self.update_weights_and_biases(self.init_weights, self.init_biases)
                        
                    # update the bias term
                    bias_copy = copy.deepcopy(self.init_biases)
                    bias_copy[layer_idx][unit_idx] = bias_copy[layer_idx][unit_idx] + 0.01
                    self.update_weights_and_biases(self.init_weights, bias_copy)
                    cur_pred = self.output(X)
                    loss = self.loss_function(y, cur_pred)
                    dJ_db = (loss - self.init_loss) / 0.01   
                    
                    # update bias parameter
                    self.biases[layer_idx][unit_idx] -= (learning_rate * dJ_db)
                    self.update_weights_and_biases(self.init_weights, self.init_biases)
           
            # Reassign weights to the layers after each EPOCH
            self.update_weights_and_biases(self.weights, self.biases)
            self.reinitialize_weights_and_biases(self.weights, self.biases)
            new_preds = self.output(X)
            preds_history.append(new_preds)  
            new_loss = self.loss_function(y, new_preds)
            if (new_loss > self.init_loss) and i > 2:
                print(f"\n{'='*40}End of Training. Convergence Reached{'='*40}\n\n")
                break 
            self.init_loss = self.loss_function(y, new_preds) 
            print(f"Loss after epoch {i+1} ******************{self.init_loss}***********************")
        if i == 99:
            print(f"\n{'='*39}End of Training. Specified EPOCHS Exhausted{'='*39}\n\n")
        return preds_history[-2], self.init_loss
