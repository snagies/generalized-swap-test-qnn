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


def generate_xor(dim, size):
    #TODO option to ensure equal frequency
    samples = []
    labels = []
    counter = 0
    for i in range(size):
        r = random.randint(0, 2**dim)
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

