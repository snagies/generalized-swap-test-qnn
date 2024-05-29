import numpy as np

from qiskit.primitives import Sampler

from algorithm.module import swap_test
from algorithm.helpers import normalize, prob_normalize

#TODO QNN composed solely of Clifford gates (except for initialize?) -> add quantum?

class ScalableQNN():
    
    def __init__(self, mod_size, mod_num = 1):
        
        self.mod_size = mod_size
        self.mod_num = mod_num
        
        self.dim = 2**self.mod_size 
        
        #TODO different weights initializations; complex weights?
        self.weights = np.zeros(self.dim * self.mod_num)
        for i in range(self.mod_num):
            self.weights[i*self.dim:(i+1)*self.dim] = normalize(np.random.normal(0, 1, self.dim))
        
        #TODO more than two labels: multiple output layer neurons -> increase coefficients
        #self.coefficients = prob_normalize(np.ones(self.mod_num))
        self.coefficients = normalize(np.random.normal(0, 1, self.mod_num))
        if self.mod_num == 1:
            self.coefficients = [1]
        
        self.loss = None
        self.gradient_weights = None
        self.gradient_coefficients = None
        
    def copy(self):
        copy = ScalableQNN(self.mod_size, self.mod_num) 
        copy.assign_weights(self.weights)
        copy.assign_coefficients(self.coefficients)
        return copy
    
    def assign_weights(self, w):
        assert len(w) == self.dim * self.mod_num
        for i in range(self.mod_num):
            self.weights[i*self.dim:(i+1)*self.dim] = normalize(w[i*self.dim:(i+1)*self.dim])
    
    def assign_coefficients(self, coeffs):
        assert len(coeffs) == self.mod_num
        self.coefficients = coeffs
    
    def update_weights(self, gradient, lr = 0.1):
        for i in range(self.mod_num):
            self.weights[i*self.dim:(i+1)*self.dim] = normalize(self.weights[i*self.dim:(i+1)*self.dim] 
                                                                - lr * gradient[i*self.dim:(i+1)*self.dim])
        #self.weights = normalize(self.weights - lr * gradient)
    
    def update_coefficients(self, gradient, lr = 0.1):
        #Necessary to normalize? allow negative coefficients
        self.coefficients = self.coefficients - lr * gradient
    
    def forward_pass(self, x, shots = None):
        #TODO option to simulate locally or on quantum hardware
        #TODO repetition of layers
        
        assert (self.dim * self.mod_num) % len(x) == 0
        
        p0 = np.zeros(self.mod_num)
        
        sampler = Sampler()
        
        for i in range(self.mod_num):
            
            weights_mod = normalize(self.weights[i*self.dim:(i+1)*self.dim])
            x_mod = normalize(x[i*self.dim:(i+1)*self.dim]) #normalizing split input vectors has effect?
            
            qc_mod = swap_test(self.mod_size, x_mod, weights_mod)
            
            #sample single module to get P(0)
            job = sampler.run(qc_mod, shots = shots)
            result = job.result()
            p0[i] = result.quasi_dists[0][0]
            
        qnn_output = np.dot(self.coefficients, p0)
        
        return qnn_output, p0
        
    def sweep(self, inputs, labels, lr = 0.05, shots = None):   
        #assume data is array of training inputs, labels is array of training labels
        #gradient is calculated purely classically
        grad_w = np.zeros(self.dim * self.mod_num)
        grad_coeffs = np.zeros(self.mod_num)
        loss = 0
        
        for i in range(len(inputs)):
            
            sample = inputs[i]
            y = labels[i]
            
            f, p = self.forward_pass(sample, shots = shots)
            
            #TODO try with sigmoid(f) for binary classification
            loss += (f - y)**2 
            
            for j in range(len(grad_coeffs)):
                grad_coeffs[j] += 2 * (f - y) * p[j]
                
            for k in range(len(grad_w)):
            
                mod_index = int(k/self.dim)
                var_index = k - mod_index * self.dim
                
                x = normalize(sample[mod_index*self.dim:(mod_index+1)*self.dim])
                w = normalize(self.weights[mod_index*self.dim:(mod_index+1)*self.dim])
                
                #grad_w[k] += x[var_index]  * np.dot(x,w) * self.coefficients[mod_index] * 2 * (f - y)
                grad_w[k] += (x[var_index]  * np.dot(x,w) - w[var_index] * np.dot(x,w)**2) * self.coefficients[mod_index] * 2 * (f - y)
             
        loss /= len(inputs)
        grad_coeffs /= len(inputs)
        grad_w /= len(inputs)
        
        self.loss = loss
        self.update_weights(grad_w, lr) 
        self.update_coefficients(grad_coeffs, lr)
        self.gradient_weights = grad_w
        self.gradient_coefficients = grad_coeffs
    
    def train(self, inputs, labels, iterations = 10, learning_rate = 0.05, shots_per_sweep = None, print_progress = False):
        
        losses = np.zeros(iterations)
        
        for i in range(iterations):
            
            self.sweep(inputs, labels, lr = learning_rate, shots = shots_per_sweep)
            losses[i] = self.loss
            
            if print_progress:
                print('Sweep ', i+1, ': Loss = ', self.loss)
                
        return losses
    
    def predict(self, inputs, labels, classes = [-1,1], shots = None, print_preds = False):
        
        if print_preds:
            print('Output   Prediction  Label')
            
        counter = 0
        
        for i in range(len(inputs)):
            f, p = self.forward_pass(inputs[i], shots)
            #TODO adapt to other possible labels
            if f > 0:
                pred = 1
            elif f < 0:
                pred = -1
            else:
                pred = 0
            
            if print_preds:
                print(f, pred, labels[i])
                
            if pred == labels[i]:
                counter += 1
        
        print(counter, ' correct predictions')
        

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    