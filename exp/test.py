import numpy as np
#import torch
from sklearn.model_selection import KFold

import sys
import os 

sys.path.insert(1, 'C:\\PhD\\code\\qml\\scalable-qnn')

from dataset.data import get_iris, generate_xor, load_mnist_binary_classifier
from algorithm.qnn import ScalableQNN, ScalableQNN_test, ScalableQNN_classical, ScalableQNN_product
from algorithm.helpers import sigmoid, normalize
import matplotlib.pyplot as plt


def expand_array_ordered(original_array, n):
    rows, cols = original_array.shape
    idx = np.arange(cols)
    idx_repeated = np.tile(idx, n)
    expanded_array = original_array[:, idx_repeated]
    
    return expanded_array


#Number of samples in the training set
num_samples = 10000
xor_dim = 1

#%%
kfold = KFold(10, shuffle = True, random_state = 1)

samples, labels = get_iris()

for train, test in kfold.split(samples):
    X_train = samples[train]
    Y_train = labels[train]
    X_test = samples[test]
    Y_test = labels[test]
    
    qnn = ScalableQNN(2, 1)
    losses = qnn.train(X_train, Y_train, iterations = 100, learning_rate = 0.05, print_progress = False)
    counter = qnn.predict(X_test, Y_test, print_preds = False)
    
    accuracy = counter / len(X_test)
    print(accuracy)




#%%
qnn = ScalableQNN(2, 1)
qnn.predict(samples, labels, print_preds = True)

losses = qnn.train(samples, labels, iterations = 400, learning_rate = 0.01, print_progress = True)
qnn.predict(samples, labels, print_preds = True)

# qnn_cl = ScalableQNN_classical(4, 1)
# qnn_cl.predict(samples, labels, print_preds = True)

# losses = qnn_cl.train(samples, labels, iterations = 2000, learning_rate = 0.01, print_progress = True)
# plt.plot(losses, '-')
# qnn_cl.predict(samples, labels, print_preds = True)
#%%
samples, labels = generate_xor(xor_dim, num_samples)  


#number of modules in network which receive copy of input
prod = 6
mods = 10
reps = prod * mods


samples_original =  np.array(samples)
samples_classical = np.copy(samples_original)

ones_column = np.ones((num_samples, 1))

#samples = np.hstack((samples_original, ones_column))


#repeat samples for each module
# samples = expand_array_ordered(samples, reps)
# samples_classical = expand_array_ordered(samples_classical, reps)


X_train, Y_train = load_mnist_binary_classifier(4, 9, quadrant_reorder=4)

#%%
#learn handwritten digits
#for single module predict is not correct, as output always positive
# x = X_train[:100,:]
# y = Y_train[:100]

x = X_train[:100,:]
y = Y_train[:100]

x2 = X_train[200:300,:]
y2 = Y_train[200:300]

#test 1 module vs 4 initialized with same weights (coefficients?)
qnn_cl = ScalableQNN_classical(196, 4)
qnn_cl.predict(x, y, print_preds = False)

losses = qnn_cl.train(x, y, iterations = 200, learning_rate = 0.05, print_progress = True)
plt.plot(losses, '-')

qnn_cl.predict(x2, y2, print_preds = False)

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

qnn_prod = ScalableQNN_product(1, prod , mods)
qnn_prod.predict(samples_classical, labels, print_preds = False)
losses = qnn_prod.train(samples_classical, labels, iterations = 3, learning_rate = 0.05, print_progress = False)
plt.plot(losses, '-')
qnn_prod.predict(samples_classical, labels, print_preds = False)
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










