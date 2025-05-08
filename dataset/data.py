import numpy as np
import os
import pathlib
import random


def get_iris(dataset = '01_iris_setosa_versicolor.csv', permute = True, size = 'all'):
    #TODO cleanup; make portable
    folderpath = pathlib.Path(__file__).parent.resolve()
    folderpath = folderpath.__str__()
    file = os.path.join(folderpath, dataset)
    
    data = np.genfromtxt(file, delimiter = ',', skip_header=1)
    
    if permute == True:
        data = np.random.permutation(data)
    
    if size == 'all':
        samples = data[:,:-1]
        labels = data[:,-1]
    #TODO ensure equal frequency in classes if size != 'all'  
    else:
        samples = data[:size,:-1]
        labels = data[:size,-1]
        
    return samples, labels


def generate_xor_old(dim, size):
    #TODO option to ensure equal frequency
    samples = []
    labels = []
    counter = 0
    for i in range(size):
        r = random.randint(0, 2**dim-1) #random.randint(0, 2**dim)
        x = []
        temp = r
        y = 0
        for d in range(dim):
            s = int((temp % 2))
            x.append((1 if s == 0 else -1) * random.random())
            y += s
            temp //= 2
        y = 1 if (y % 2) == 0 else -1
        if y == 1:
            counter += 1
        samples.append(x)
        labels.append(y)
    print(counter, ' instances with label +1')
    return samples, labels


def generate_xor(dim, size):
    #new version
    samples = 2 * np.random.random(size=(size, dim)) - 1
    labels = np.sign(np.prod(samples, axis=1))
    counter = np.sum(labels == 1)
    print(counter, ' instances with label +1')
    return samples, labels

def generate_xor_balanced(dim, n_samples_dim=1000, shuffle=True):
    samples = np.random.random(size=(2**dim*n_samples_dim, dim))
    for i in range(2**dim):
        signs = np.array([1 if int((i // 2**d) % 2) == 0 else -1 for d in range(dim)])
        samples[i*n_samples_dim:(i+1)*n_samples_dim] *= signs
    labels = np.sign(np.prod(samples, axis=1))
    if shuffle:
        perm = np.random.permutation(2**dim*n_samples_dim)
        samples = samples[perm]
        labels = labels[perm]
    return samples, labels

def generate_spirals(n_samples_class=1000, noise=0.025, n_rounds=3, shuffle=True):
    theta = np.linspace(1, n_rounds * 2 * np.pi, n_samples_class)
    r = theta * 0.1
    x_0 = r * np.sin(theta) + noise * np.random.randn(n_samples_class)
    y_0 = r * np.cos(theta) + noise * np.random.randn(n_samples_class)
    x_1 = -r * np.sin(theta) + noise * np.random.randn(n_samples_class)
    y_1 = -r * np.cos(theta) + noise * np.random.randn(n_samples_class)
    x = np.concatenate((x_0, x_1))
    y = np.concatenate((y_0, y_1))
    labels = np.ones(2 * n_samples_class)
    labels[:n_samples_class] = -1
    samples = np.column_stack((x, y))
    if shuffle:
        perm = np.random.permutation(2 * n_samples_class)
        samples = samples[perm]
        labels = labels[perm]
    return samples, labels