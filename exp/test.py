import numpy as np

from dataset.data import get_iris
from algorithm.qnn import ScalableQNN

samples, labels = get_iris()

qnn = ScalableQNN(1,2)

losses = qnn.train(samples, labels, iterations = 100, learning_rate = 0.1,
                   print_progress = True)

print(qnn.coefficients)

qnn.predict(samples, labels)