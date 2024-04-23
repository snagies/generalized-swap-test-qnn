import numpy as np


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

def cosine_sim(a, b):
    x = normalize(a)
    y = normalize(b)
    return np.dot(x, y)

def euclidean_dist(a, b):
    return np.linalg.norm(a - b)