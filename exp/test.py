import numpy as np

from dataset.data import get_iris, generate_xor
from algorithm.qnn import ScalableQNN
from algorithm.helpers import sigmoid
import matplotlib.pyplot as plt

import sys
import os 

sys.path.insert(1, 'C:\\PhD\\code\\qml\\scalable-qnn')

samples, labels = get_iris()
#samples, labels = generate_xor(4,100)
samples_train = samples[50:]
samples_test = samples[:50]

qnn = ScalableQNN(1,2)
qnn_copy = qnn.copy()

losses = qnn.train(samples, labels, iterations = 100, learning_rate = 0.1, shots_per_sweep = None,
                   print_progress = True)

#print(qnn.coefficients)

qnn.predict(samples, labels, print_preds = True)



x = np.linspace(0,1.5,100)
y, der = sigmoid(x, a = 5)
plt.plot(x, y,'.')
plt.plot(x, der,'.')