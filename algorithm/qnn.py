import numpy as np
#from qiskit import transpile
from qiskit.primitives import Sampler
from module import swap_test
from helpers import normalize, prob_normalize

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
        
        self.coefficients = prob_normalize(np.ones(self.mod_num))
        
        self.loss = None
        self.gradient_weights = None
        self.gradient_coefficients = None
        
    
    def assign_weights(self, w):
        assert len(w) == self.dim * self.mod_num
        for i in range(self.mod_num):
            self.weights[i*self.dim:(i+1)*self.dim] = normalize(w[i*self.dim:(i+1)*self.dim])
    
    def assign_coefficients(self, coeffs):
        assert len(coeffs) == self.mod_num
        self.coefficients = prob_normalize(coeffs)
    
    def update_weights(self, gradient, learning_rate = 0.1):
        self.weights = normalize(self.weights - learning_rate * gradient)
    
    def update_coefficients(self, gradient, learning_rate = 0.1):
        self.coefficients = prob_normalize(self.coefficients - learning_rate * gradient)
    
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
        
    def sweep(self, inputs, labels, lr = 0.01, shots = None):   
        #assume data is array of training inputs, labels is array of training labels
        grad_w = np.zeros(self.dim * self.mod_num)
        grad_coeffs = np.zeros(self.mod_num)
        loss = 0
        
        for i in range(len(inputs)):
            sample = inputs[i]
            y = labels[i]
            
            f, p = self.forward_pass(sample, shots = shots)
            
            loss += (f - y)**2 / len(inputs)
            
            for j in range(len(grad_coeffs)):
                grad_coeffs[j] += 2 * (f - y) * p[j] / len(inputs)
                
            for k in range(len(grad_w)):
                #TODO implement
                grad_w += 0 / len(inputs)
                
        self.loss = loss
        self.update_weights(grad_w, lr) 
        self.update_coefficients(grad_coeffs, lr)
        self.gradient_weights = grad_w
        self.gradient_coefficients = grad_coeffs
    
    def train(self, inputs, labels, iterations = 1, shots_per_sweep = None):
        for i in range(iterations):
            self.sweep(inputs, labels)
            print('Sweep ', i+1, ': Loss = ', self.loss)

    
    
    
    
#test = normalize(np.random.rand(4))
#w = normalize(np.random.rand(4))    

#w = normalize(w - test*np.dot(test,w))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    