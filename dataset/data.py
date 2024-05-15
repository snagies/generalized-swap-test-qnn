import numpy as np
import os
import pathlib


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

