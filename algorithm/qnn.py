import numpy as np
from qiskit_machine_learning.circuit.library import QNNCircuit
from qiskit import transpile
from qiskit.providers.basic_provider import BasicProvider
from module import swap_test
from helpers import normalize


class ScalableQNN():
    
    def __init__(self, mod_size, mod_num = 1):
        
        self.mod_size = mod_size
        self.mod_num = mod_num
        
        self.dim = 2**self.mod_size 
        
        #TODO different weights initializations
        self.weights = normalize(np.random.normal(0, 1, self.dim * self.mod_num))
        self.coefficients = normalize(np.ones(self.mod_num))
        
        self.loss = None
        self.output = None
        
        self.fm, self.ae = swap_test(self.mod_size, fm_explicit = True)
        
    def update_weights(self, data, learning_rate = 0.1):
        gradient = 0
        self.weights = normalize(self.weights - learning_rate * gradient)
    
    def squared_loss(self, data):
        loss = 0
        self.loss = loss
        
    def forward_pass(self, x, shots = 100):
        #calculate qnn output for one sample (fitting completely onto modules)
        
        assert (self.dim * self.mod_num) % len(x) == 0
        
        p0 = np.zeros(self.mod_num)
        
        for i in range(self.mod_num):
            
            self.fm, self.ae = swap_test(self.mod_size, fm_explicit = True)
            
            self.fm.assign_parameters(normalize(x[i*self.dim:(i+1)*self.dim]), inplace = True) 
            self.ae.assign_parameters(self.weights[i*self.dim:(i+1)*self.dim], inplace = True)
            
            self.qnn = QNNCircuit(feature_map = self.fm, ansatz = self.ae)
            
            #TODO sample module to get P(0)
            #...
            p0[i] = 0.5
            
        sample_output = np.dot(self.coefficients, p0)
        
        return sample_output, p0
        
        
        
        #provider = BasicProvider()
        #backend = provider.get_backend("basic_simulator")
        
        #trans_qc = transpile(self.qnn, backend)
        