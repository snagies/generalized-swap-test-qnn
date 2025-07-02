import numpy as np

from qiskit.primitives import Sampler

from algorithm.module import swap_test
from algorithm.helpers import normalize, prob_normalize, sigmoid


class SWAP_test_QNN():
    
    """
    An implementation of a QNN based on SWAP tests, corresponding to a classical two-layer
    neural network with quadratic activation functions.
    
    Parameters:
    -----------
    mod_size : int
        Size parameter determining module dimension (2^mod_size), corresponding to number
        of qubits in each register
    mod_num : int, default=1
        Number of SWAP tests in the network
        
    Attributes:
    -----------
    dim : int
        Dimension of each module (2^mod_size)
    weights : numpy.ndarray
        Flattened array of all module weights (normalized)
    coefficients : numpy.ndarray
        Classical coefficients in second layer
    loss : float
        Last computed training loss
    gradient_weights : numpy.ndarray
        Last computed weight gradients
    gradient_coefficients : numpy.ndarray
        Last computed coefficient gradients
        
    Methods:
    --------
    forward_pass(x, shots=None) : tuple
        Compute network output and module probabilities
    train(inputs, labels, iterations=10, learning_rate=0.05, ...) : numpy.ndarray
        Train the network and return loss history
    predict(inputs, labels, classes=[-1,1], ...) : int
        Make predictions on test dataset
    assign_weights(w, module_index=None) : None
        Update network weights (all or specific module)
    assign_coefficients(coeffs, module_index=None) : None
        Update module coefficients without normalization constraints
        
    Example:
    --------
    >>> qnn = SWAP_test_QNN(mod_size=3, mod_num=2)  # 2 modules of 8 dimensions each
    >>> losses = qnn.train(X_train, y_train, iterations=100)
    >>> accuracy = qnn.predict(X_test, y_test, print_preds=True)
    
    """
    
    def __init__(self, mod_size, mod_num = 1):
        
        self.mod_size = mod_size
        self.mod_num = mod_num
        
        self.dim = 2**self.mod_size 
        
        self.weights = np.zeros(self.dim * self.mod_num)
        for i in range(self.mod_num):
            self.weights[i*self.dim:(i+1)*self.dim] = normalize(np.random.normal(0, 1, self.dim))
        
        self.coefficients = np.random.normal(0, 0.1, mod_num)
        if self.mod_num == 1:
            self.coefficients = [1]
        
        self.loss = None
        self.gradient_weights = None
        self.gradient_coefficients = None
       
        
    def copy(self):
        copy = SWAP_test_QNN(self.mod_size, self.mod_num) 
        copy.assign_weights(self.weights)
        copy.assign_coefficients(self.coefficients)
        return copy
    
    
    def assign_weights_old(self, w):
        assert len(w) == self.dim * self.mod_num
        for i in range(self.mod_num):
            self.weights[i*self.dim:(i+1)*self.dim] = normalize(w[i*self.dim:(i+1)*self.dim])
    
    
    def assign_weights(self, w, module_index=None):
        """
        Assign new weights to the network.
    
        Parameters:
        w (numpy.ndarray): New weights to assign.
        module_index (int, optional): If provided, only update weights for the specified module.
        """
        if module_index is not None:
            if module_index < 0 or module_index >= self.mod_num:
                raise ValueError(f"Invalid module index. Must be between 0 and {self.mod_num - 1}.")
            if len(w) != self.dim:
                raise ValueError(f"Weight vector length must be {self.dim} for a single module.")
        
            start_index = module_index * self.dim
            end_index = start_index + self.dim
            self.weights[start_index:end_index] = normalize(w)
        else:
            assert len(w) == self.dim * self.mod_num, "Weight vector length must match total number of weights."
            for i in range(self.mod_num):
                start_index = i * self.dim
                end_index = start_index + self.dim
                self.weights[start_index:end_index] = normalize(w[start_index:end_index])
    
    
    def assign_coefficients_old(self, coeffs):
        assert len(coeffs) == self.mod_num
        self.coefficients = coeffs
    
    
    def assign_coefficients(self, coeffs, module_index=None):
        """
        Assign new coefficients to the network.
    
        Parameters:
        coeffs (numpy.ndarray or float): New coefficient(s) to assign.
        module_index (int, optional): If provided, only update the coefficient for the specified module.
        """
        if module_index is not None:
            if module_index < 0 or module_index >= self.mod_num:
                raise ValueError(f"Invalid module index. Must be between 0 and {self.mod_num - 1}.")
            if not isinstance(coeffs, (int, float)):
                raise ValueError("For a single module, coefficient must be a scalar value.")
        
            self.coefficients[module_index] = coeffs
        else:
            assert len(coeffs) == self.mod_num, "Coefficient vector length must match number of modules."
            self.coefficients = coeffs
    
    
    def update_weights(self, gradient, lr = 0.1):
        for i in range(self.mod_num):
            self.weights[i*self.dim:(i+1)*self.dim] = normalize(self.weights[i*self.dim:(i+1)*self.dim] 
         
                                                                - lr * gradient[i*self.dim:(i+1)*self.dim])
    def update_coefficients(self, gradient, lr = 0.1):
        self.coefficients = self.coefficients - lr * gradient
    
    
    def print_module_weights(self, module_index):
        """
        Print the current weights within a specified module.
    
        Parameters:
        module_index (int): The index of the module to print weights for (0-indexed).
        """
        if module_index < 0 or module_index >= self.mod_num:
            print(f"Error: Invalid module index. Must be between 0 and {self.mod_num - 1}.")
            return
    
        start_index = module_index * self.dim
        end_index = start_index + self.dim
        module_weights = self.weights[start_index:end_index]
    
        print(f"Weights for module {module_index}:")
        for i, weight in enumerate(module_weights):
            print(f"  Weight {i}: {weight:.6f}")
        print(f"Module coefficient: {self.coefficients[module_index]:.6f}")
    
    def forward_pass(self, x, shots = None):
        
        assert (self.dim * self.mod_num) % len(x) == 0
        
        p0 = np.zeros(self.mod_num)
 
        
        for i in range(self.mod_num):
            
            weights_mod = normalize(self.weights[i*self.dim:(i+1)*self.dim])
            x_mod = normalize(x[i*self.dim:(i+1)*self.dim]) 
    
            p0[i] = 0.5 * (1 + np.abs(np.vdot(x_mod,weights_mod))**2)
            
        qnn_output = np.dot(self.coefficients, p0)
        
        return qnn_output, p0
        
    def sweep(self, inputs, labels, lr = 0.05, shots = None, a = 3):
     
        grad_w = np.zeros(self.dim * self.mod_num)
        grad_coeffs = np.zeros(self.mod_num)
        loss = 0
        
        for i in range(len(inputs)):
            
            sample = inputs[i]
            y = labels[i]
            
            f, p = self.forward_pass(sample, shots = shots)
            
            sig, sig_der = sigmoid(f, a , c=2.5)
            
            loss += (sig - y)**2 
            
            for j in range(len(grad_coeffs)):
                grad_coeffs[j] += 2 * (sig - y) * p[j] * sig_der
                
            for k in range(len(grad_w)):
            
                mod_index = int(k/self.dim)
                var_index = k - mod_index * self.dim
                
                x = normalize(sample[mod_index*self.dim:(mod_index+1)*self.dim])
                w = normalize(self.weights[mod_index*self.dim:(mod_index+1)*self.dim])
                
                grad_w[k] += ((x[var_index]  * np.vdot(x,w) - w[var_index] * np.vdot(x,w)**2) 
                              * self.coefficients[mod_index] * 2 * (sig - y) * sig_der)   
             
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

            if f > 0.75:
                pred = 1
            elif f < 0.75:
                pred = -1
            else:
                pred = 0
            
            if print_preds:
                print(sigmoid(f, a = 3,b =1, c=0, d= 0.5, return_der = False), pred, labels[i])
                
            if pred == labels[i]:
                counter += 1
        
        print(counter, ' correct predictions')     



       
    
class SWAP_test_QNN_classical():
    
    """
    Classical variant of SWAP_test_QNN, with the bias now explicitly given as an additional trainable parameter
    """
    
    def __init__(self, mod_size, mod_num = 1):
        
        self.mod_size = mod_size
        self.mod_num = mod_num
        
        self.dim = self.mod_size
        
        self.weights = np.zeros(self.dim * self.mod_num)
        for i in range(self.mod_num):
            self.weights[i*self.dim:(i+1)*self.dim] = normalize(np.random.normal(0, 1, self.dim))
        
        self.coefficients = np.random.normal(0, 0.1, self.mod_num)
        if self.mod_num == 1:
            self.coefficients = [1]
        
        self.bias = np.random.normal(0, 0.1, self.mod_num)
        
        self.loss = None
        self.gradient_weights = None
        self.gradient_coefficients = None
        self.gradient_bias = None
       
        
    def copy(self):
        copy = SWAP_test_QNN_classical(self.mod_size, self.mod_num) 
        copy.assign_weights(self.weights)
        copy.assign_coefficients(self.coefficients)
        copy.assign_bias(self.bias)
        return copy
    
    def assign_bias(self, b):
        assert len(b) == self.mod_num
        for i in range(self.mod_num):
            self.bias = b
    
    def assign_weights(self, w, module_index=None):
        """
        Assign new weights to the network.
    
        Parameters:
        w (numpy.ndarray): New weights to assign.
        module_index (int, optional): If provided, only update weights for the specified module.
        """
        if module_index is not None:
            if module_index < 0 or module_index >= self.mod_num:
                raise ValueError(f"Invalid module index. Must be between 0 and {self.mod_num - 1}.")
            if len(w) != self.dim:
                raise ValueError(f"Weight vector length must be {self.dim} for a single module.")
        
            start_index = module_index * self.dim
            end_index = start_index + self.dim
            self.weights[start_index:end_index] = normalize(w)
        else:
            assert len(w) == self.dim * self.mod_num, "Weight vector length must match total number of weights."
            for i in range(self.mod_num):
                start_index = i * self.dim
                end_index = start_index + self.dim
                self.weights[start_index:end_index] = normalize(w[start_index:end_index])
    
    def assign_coefficients(self, coeffs, module_index=None):
        """
        Assign new coefficients to the network.
    
        Parameters:
        coeffs (numpy.ndarray or float): New coefficient(s) to assign.
        module_index (int, optional): If provided, only update the coefficient for the specified module.
        """
        if module_index is not None:
            if module_index < 0 or module_index >= self.mod_num:
                raise ValueError(f"Invalid module index. Must be between 0 and {self.mod_num - 1}.")
            if not isinstance(coeffs, (int, float)):
                raise ValueError("For a single module, coefficient must be a scalar value.")
        
            self.coefficients[module_index] = coeffs
        else:
            assert len(coeffs) == self.mod_num, "Coefficient vector length must match number of modules."
            self.coefficients = coeffs
    
    def update_weights(self, gradient, lr = 0.1):
        for i in range(self.mod_num):
            self.weights[i*self.dim:(i+1)*self.dim] = normalize(self.weights[i*self.dim:(i+1)*self.dim] 
                                                                - lr * gradient[i*self.dim:(i+1)*self.dim])
            
    def update_coefficients(self, gradient, lr = 0.1):
        self.coefficients = self.coefficients - lr * gradient
        
    def update_bias(self, gradient, lr = 0.1):
        self.bias = self.bias - lr * gradient    
    
    def print_module_weights(self, module_index):
        """
        Print the current weights within a specified module.
    
        Parameters:
        module_index (int): The index of the module to print weights for (0-indexed).
        """
        if module_index < 0 or module_index >= self.mod_num:
            print(f"Error: Invalid module index. Must be between 0 and {self.mod_num - 1}.")
            return
    
        start_index = module_index * self.dim
        end_index = start_index + self.dim
        module_weights = self.weights[start_index:end_index]
    
        print(f"Weights for module {module_index}:")
        for i, weight in enumerate(module_weights):
            print(f"  Weight {i}: {weight:.6f}")
        print(f"Module bias: {self.bias[module_index]:.6f}")
        print(f"Module coefficient: {self.coefficients[module_index]:.6f}")
    
    def forward_pass(self, x):
        
        assert (self.dim * self.mod_num) % len(x) == 0
        
        p0 = np.zeros(self.mod_num)
        
        pre_act = np.zeros(self.mod_num)
        act = np.zeros(self.mod_num)
        act_der = np.zeros(self.mod_num)
        
        for i in range(self.mod_num):
            
            weights_mod = normalize(self.weights[i*self.dim:(i+1)*self.dim])
            x_mod = normalize(x[i*self.dim:(i+1)*self.dim]) 
            
            pre_act[i] = np.dot(x_mod, weights_mod) + self.bias[i]
            
            act[i], act_der[i] = pre_act[i]**2, 2 * pre_act[i]

            p0[i] = act[i]
            
        qnn_output = np.dot(self.coefficients, p0)
        
        return qnn_output, p0, act_der
        
    def sweep(self, inputs, labels, lr = 0.05):
        
        grad_w = np.zeros(self.dim * self.mod_num)
        grad_coeffs = np.zeros(self.mod_num)
        grad_bias = np.zeros(self.mod_num)
        loss = 0
        
        for i in range(len(inputs)):
            
            sample = inputs[i]
            y = labels[i]
            
            f, m, deriv = self.forward_pass(sample)
            
            loss += (f - y)**2 
            
            for j in range(len(grad_coeffs)):
                grad_coeffs[j] += 2 * (f - y) * m[j] 
                
            for l in range(len(grad_bias)):
                x = normalize(sample[l*self.dim:(l+1)*self.dim])
                w = normalize(self.weights[l*self.dim:(l+1)*self.dim])
                
                grad_bias[l] += 2 * (f - y) * deriv[l] * self.coefficients[l]
                
            for k in range(len(grad_w)):
            
                mod_index = int(k/self.dim)
                var_index = k - mod_index * self.dim
                
                x = normalize(sample[mod_index*self.dim:(mod_index+1)*self.dim])
                w = normalize(self.weights[mod_index*self.dim:(mod_index+1)*self.dim])
            
                grad_w[k] += 2 * (f - y) * self.coefficients[mod_index] * deriv[mod_index] * ( x[var_index] - np.dot(x,w) * w[var_index] )
             
        loss /= len(inputs)
        grad_coeffs /= len(inputs)
        grad_bias /= len(inputs)
        grad_w /= len(inputs)
        
        self.loss = loss
        self.update_weights(grad_w, lr) 
        self.update_coefficients(grad_coeffs, lr)
        self.update_bias(grad_bias, lr)
        self.gradient_weights = grad_w
        self.gradient_coefficients = grad_coeffs
        self.gradient_bias = grad_bias
        
        
    def train(self, inputs, labels, iterations = 100, learning_rate = 0.05, print_progress = False):
        
        losses = np.zeros(iterations)
        
        for i in range(iterations):
            
            self.sweep(inputs, labels, lr = learning_rate)
            losses[i] = self.loss
            
            if print_progress:
                print('Sweep ', i+1, ': Loss = ', self.loss)
                
        return losses
    
    def predict(self, inputs, labels, print_preds = False):
        
        if print_preds:
            print('Output   Prediction  Label')
            
        counter = 0
        
        for i in range(len(inputs)):
            f, p, _ = self.forward_pass(inputs[i])
            
            if f > 0:
                pred = +1
            elif f < 0:
                pred = -1
            else:
                pred = 0
          
            if print_preds:
                print(f, pred, labels[i])
                
            if pred == labels[i]:
                counter += 1
        
        print(counter, ' correct predictions')    
    
    

class SWAP_test_QNN_product():
    """
    Generalization of SWAP_test_QNN, implementing the analogue of a product layer
    """
    
    def __init__(self, mod_size, mod_prod = 1, mod_num = 1):
        
        self.mod_size = mod_size
        self.mod_prod = mod_prod
        self.mod_num = mod_num
        
        self.dim = self.mod_size

        self.weights = np.zeros((self.mod_num, self.mod_prod, self.dim))
        for i in range(self.mod_num):
            for j in range(self.mod_prod):
                self.weights[i,j,:] = normalize(np.random.normal(0, 1, self.dim))
 
        self.coefficients = np.random.normal(0, 0.1, self.mod_num)
        if self.mod_num == 1:
            self.coefficients = [1]
        
        self.bias = np.random.normal(0, 0.1, (self.mod_num, self.mod_prod))
        
        self.loss = None
        self.gradient_weights = None
        self.gradient_coefficients = None
        self.gradient_bias = None    
        
    def update_weights(self, gradient, lr = 0.1):
        for i in range(self.mod_num):
            for j in range(self.mod_prod):
                self.weights[i,j,:] = normalize(self.weights[i,j,:] - lr * gradient[i,j,:])
                   
    def update_coefficients(self, gradient, lr = 0.1):
        self.coefficients = self.coefficients - lr * gradient
        
    def update_bias(self, gradient, lr = 0.1):
        for i in range(self.mod_num):
            self.bias[i,:] = self.bias[i,:] - lr * gradient[i,:]
            
    def forward_pass(self, x):
        assert self.dim == len(x)
        
        x = normalize(x)
        
        p0 = np.ones(self.mod_num)
        
        pre_act = np.zeros((self.mod_num, self.mod_prod))
        act = np.zeros((self.mod_num, self.mod_prod))
        act_der = np.zeros((self.mod_num, self.mod_prod))
        
        
        
        for i in range(self.mod_num):
            for j in range(self.mod_prod):
                weights_mod = self.weights[i, j, :]
                
                pre_act[i, j] = np.dot(x, weights_mod) + self.bias[i, j]
                pre_act[i, j] = np.clip(pre_act[i, j], -100, 100)
                
                act[i, j], act_der[i, j] = pre_act[i, j]**2, 2 * pre_act[i, j]
                
                p0[i] *= act[i,j]
            
        qnn_output = np.dot(self.coefficients, p0)
        
        return qnn_output, p0, act_der
    
    def sweep(self, inputs, labels, lr = 0.05):

        grad_w = np.zeros((self.mod_num, self.mod_prod, self.dim))
        grad_coeffs = np.zeros(self.mod_num)
        grad_bias = np.zeros((self.mod_num, self.mod_prod))
        loss = 0
        
        epsilon = 1e-10
        
        for i in range(len(inputs)):
            
            sample = normalize(inputs[i])
            y = labels[i]
            
            f, m, deriv = self.forward_pass(sample)
            
            loss += (f - y)**2 
            
            for j in range(self.mod_num):
                grad_coeffs[j] += 2 * (f - y) * m[j] 
                
            for k in range(self.mod_num):
                for l in range(self.mod_prod):
                    grad_bias[k,l] += 2 * (f - y) * m[k] * (np.dot(sample, self.weights[k,l,:]) + self.bias[k,l] + epsilon)**(-1) * 2 * self.coefficients[k]
                     

            for k in range(self.mod_num):
                for l in range(self.mod_prod):
                    for n in range(self.dim):
                        grad_w[k,l,n] += 2 * (f - y) * m[k] * (np.dot(sample, self.weights[k,l,:]) + self.bias[k,l] + epsilon)**(-1) * 2 * self.coefficients[k] * (sample[n] - np.dot(sample, self.weights[k,l,:]) * self.weights[k,l,n])     
                    
             
        loss /= len(inputs)
        grad_coeffs /= len(inputs)
        grad_bias /= len(inputs)
        grad_w /= len(inputs)
        
        self.loss = loss
        self.update_weights(grad_w, lr) 
        self.update_coefficients(grad_coeffs, lr)
        self.update_bias(grad_bias, lr)
        self.gradient_weights = grad_w
        self.gradient_coefficients = grad_coeffs
        self.gradient_bias = grad_bias
        
    def train(self, inputs, labels, iterations = 100, learning_rate = 0.05, print_progress = False):
        
        losses = np.zeros(iterations)
        
        for i in range(iterations):
            
            self.sweep(inputs, labels, lr = learning_rate)
            losses[i] = self.loss
            
            if print_progress:
                print('Sweep ', i+1, ': Loss = ', self.loss)
                
        return losses
                
    def predict(self, inputs, labels, print_preds = False):
        
        if print_preds:
            print('Output   Prediction  Label')
            
        counter = 0
        
        for i in range(len(inputs)):
            f, p, _ = self.forward_pass(inputs[i])
            
            if f > 0:
                pred = +1
            elif f < 0:
                pred = -1
            else:
                pred = 0
          
            if print_preds:
                print(f, pred, labels[i])
                
            if pred == labels[i]:
                counter += 1
        
        print(counter, ' correct predictions')
    
    
    
            
        
    
    
    