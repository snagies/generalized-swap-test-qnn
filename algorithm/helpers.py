import numpy as np
from itertools import combinations


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

def prob_normalize(v):
    assert v.min() >= 0
    prob_sum = np.sum(v)
    return v / prob_sum

def cosine_sim(a, b):
    x = normalize(a)
    y = normalize(b)
    return np.dot(x, y)

def euclidean_dist(a, b):
    return np.linalg.norm(a - b)

def sigmoid(x, a = 10, b = 2, c = 0.75, d = 1, return_der = True):
    y = (b / (1 + np.exp(-a * (x-c)))) - d
    der = a * b * np.exp(-a * (x-c)) / (np.exp(-a * (x-c)) + 1)**2
    if return_der:
        return y, der
    else: 
        return y
    
def n_choose_k(n, k):
    if k <= 0 or k > n:
        return ""
    return ','.join('-'.join(map(str, combo)) for combo in combinations(range(n), k))
    


