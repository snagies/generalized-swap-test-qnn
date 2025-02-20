import numpy as np

import sys
import os 

#sys.path.insert(1, 'C:\\PhD\\code\\qml\\scalable-qnn')

from dataset.data import get_iris, generate_xor
from algorithm.qnn import ScalableQNN, ScalableQNN_test, ScalableQNN_classical
from algorithm.helpers import sigmoid, normalize
import matplotlib.pyplot as plt


def expand_array_ordered(original_array, n):
    rows, cols = original_array.shape
    idx = np.arange(cols)
    idx_repeated = np.tile(idx, n)
    expanded_array = original_array[:, idx_repeated]
    
    return expanded_array


#Number of samples in the training set
num_samples = 100
xor_dim = 3


#generate data (XOR or iris)
#samples, labels = get_iris()
samples, labels = generate_xor(xor_dim, num_samples)  


#number of modules in network which receive copy of input
reps = 8


samples_original =  np.array(samples)
samples_classical = np.copy(samples_original)

ones_column = np.ones((num_samples, 1))

#samples = np.hstack((samples_original, ones_column))


#repeat samples for each module
#samples = expand_array_ordered(samples, reps)
samples_classical = expand_array_ordered(samples_classical, reps)


#%%
#Initialize NN with reps modules with 3 input dimensions each
qnn_cl = ScalableQNN_classical(3, reps)

#Number of correctly predicted samples for randomly initialized network
qnn_cl.predict(samples_classical, labels, print_preds = False)

#train network with quadratic loss function and gradient descent
losses = qnn_cl.train(samples_classical, labels, iterations = 1500, learning_rate = 0.1, print_progress = False)
plt.plot(losses, '-')

#predicitions after training
qnn_cl.predict(samples_classical, labels, print_preds = False)

#%%


# qnn = ScalableQNN_test(2,reps)
# qnn.assign_coefficients([1]*reps)
# #qnn_copy = qnn.copy()
#%%
# qnn_xor = ScalableQNN_test(2,8)

# ### assign coefficients
# qnn_xor.assign_coefficients([1,1,1,1,  0,0,0,0])


# ### assign weights
# qnn_xor.assign_weights([ 1, 1, 1,   1],  0)
# qnn_xor.assign_weights([-1,-1,-1,   -1],  1)

# qnn_xor.assign_weights([ 1,-1,-1,   1],  2)
# qnn_xor.assign_weights([-1, 1, 1,   1],  3)

# qnn_xor.assign_weights([-1, 1,-1,   1],  4)
# qnn_xor.assign_weights([ 1,-1, 1,   1],  5)

# qnn_xor.assign_weights([-1,-1, 1,   1],  6)
# qnn_xor.assign_weights([ 1, 1,-1,   1],  7)


# ### test output of QNN
# output, p = qnn_xor.forward_pass([ 1, 1, 1,  1]*8);    print('Label:', '+1','Output:',output,p)
# output, p = qnn_xor.forward_pass([-1,-1,-1,  1]*8);    print('Label:', '-1','Output:',output,p)

# output, p = qnn_xor.forward_pass([ 1,-1,-1,  1]*8);    print('Label:', '+1','Output:',output,p)
# output, p = qnn_xor.forward_pass([-1, 1, 1,  1]*8);    print('Label:', '-1','Output:',output,p)

# output, p = qnn_xor.forward_pass([-1, 1,-1,  1]*8);    print('Label:', '+1','Output:',output,p)
# output, p = qnn_xor.forward_pass([ 1,-1, 1,  1]*8);    print('Label:', '-1','Output:',output,p)

# output, p = qnn_xor.forward_pass([-1,-1, 1,  1]*8);    print('Label:', '+1','Output:',output,p)
# output, p = qnn_xor.forward_pass([ 1, 1,-1,  1]*8);    print('Label:', '-1','Output:',output,p)




# #print(qnn_xor.coefficients)
# # qnn_xor.print_module_weights(0)
# # qnn_xor.print_module_weights(1)
# # qnn_xor.print_module_weights(2)
# # qnn_xor.print_module_weights(3)

#%%


# ### assign coefficients
# qnn_cl.assign_coefficients([1,1,1,1,  1,1,1,1])

# ### assign biases
# qnn_cl.assign_bias([ 0.  , -0.29,  0.6 ,  0.02,  0.27,  0.35, -0.54,  0.25 ])  #-1/np.sqrt(3)

# ### assign weights
# qnn_cl.assign_weights([ 1, 1, 1],  7)
# qnn_cl.assign_weights([-1,-1,-1],  0)

# qnn_cl.assign_weights([ 1,-1,-1],  4)
# qnn_cl.assign_weights([-1, 1, 1],  3)

# qnn_cl.assign_weights([-1, 1,-1],  2)
# qnn_cl.assign_weights([ 1,-1, 1],  5)

# qnn_cl.assign_weights([-1,-1, 1],  1)
# qnn_cl.assign_weights([ 1, 1,-1],  6)





### test output of QNN
# output, p, _  = qnn_cl.forward_pass([ 1, 1, 1]*8);    print('Label:', '+1','Output:',round(output,2),p)
# output, p, _ = qnn_cl.forward_pass([-1,-1,-1]*8);    print('Label:', '-1','Output:',round(output,2),p)
 
# output, p, _ = qnn_cl.forward_pass([ 1,-1,-1]*8);    print('Label:', '+1','Output:',round(output,2),p)
# output, p, _ = qnn_cl.forward_pass([-1, 1, 1]*8);    print('Label:', '-1','Output:',round(output,2),p)

# output, p, _ = qnn_cl.forward_pass([-1, 1,-1]*8);    print('Label:', '+1','Output:',round(output,2),p)
# output, p, _ = qnn_cl.forward_pass([ 1,-1, 1]*8);    print('Label:', '-1','Output:',round(output,2),p)

# output, p, _ = qnn_cl.forward_pass([-1,-1, 1]*8);    print('Label:', '+1','Output:',round(output,2),p)
# output, p, _ = qnn_cl.forward_pass([ 1, 1,-1]*8);    print('Label:', '-1','Output:',round(output,2),p)


# qnn_cl.print_module_weights(0)
# qnn_cl.print_module_weights(1)
# qnn_cl.print_module_weights(2)
# qnn_cl.print_module_weights(3)
# qnn_cl.print_module_weights(4)
# qnn_cl.print_module_weights(5)
# qnn_cl.print_module_weights(6)
# qnn_cl.print_module_weights(7)

# losses = qnn_cl.train(samples_classical, labels, iterations = 1000, learning_rate = 0.1, print_progress = False)
# plt.plot(losses, '-')
# qnn_cl.predict(samples_classical, labels, print_preds = False)
#%%
#test ability to learn xor for 1 module
# qnn_xor = ScalableQNN_test(2,1)
# qnn_xor.assign_weights([1,1,1,2])
# output1, _ = qnn_xor.forward_pass([1,1,1,1])
# output2, _ = qnn_xor.forward_pass([-1,-1,-1,1])
# print(output1, output2)

# qnn_xor.assign_weights([-1,-1,1,2])
# output1, _ = qnn_xor.forward_pass([-1,-1,1,1])
# output2, _ = qnn_xor.forward_pass([1,1,-1,1])
# print(output1, output2)

# qnn_xor.assign_weights([1,-1,-1,2])
# output1, _ = qnn_xor.forward_pass([1,-1,-1,1])
# output2, _ = qnn_xor.forward_pass([-1,1,1,1])
# print(output1, output2)

# qnn_xor.assign_weights([-1,1,-1,2])
# output1, _ = qnn_xor.forward_pass([-1,1,-1,1])
# output2, _ = qnn_xor.forward_pass([1,-1,1,1])
# print(output1, output2)


#%%
#x = qnn.weights[:4]
#y = qnn.weights[4:]
#new_weights = normalize(x - y * np.dot(x,y))
#new_weights = np.append(new_weights,y)

#qnn.assign_weights(new_weights)



#plt.plot(qnn.weights[0],qnn.weights[1],'.',color='green', markersize = 15)

# losses = qnn.train(samples, labels, iterations = 2000, learning_rate = 0.5, shots_per_sweep = None,
#                     print_progress = True)

# losses = qnn_xor.train(samples, labels, iterations = 2000, learning_rate = 0.1, shots_per_sweep = None,
#                    print_progress = True)
# #print(qnn.coefficients)

#plt.plot(losses, 'r.')

#plt.plot(qnn.weights[0],qnn.weights[1],'.',color='green', markersize = 15)

#qnn.predict(samples, labels, print_preds = False)
#qnn.predict(samples_test, labels_test, print_preds = False)










