import numpy as np


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


def sigmoid(x, a = 5):
    y = (2 / (1 + np.exp(-a * x))) - 1
    return y