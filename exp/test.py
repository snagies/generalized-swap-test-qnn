import numpy as np

from dataset.data import get_iris, generate_xor
from algorithm.qnn import ScalableQNN

import sys
import os 

sys.path.insert(1, 'C:\\PhD\\code\\qml\\scalable-qnn')


samples, labels = get_iris()
#samples, labels = generate_xor(4,100)
samples_train = samples[50:]

qnn = ScalableQNN(1,2)

losses = qnn.train(samples, labels, iterations = 100, learning_rate = 0.5, shots_per_sweep = None,
                   print_progress = True)

print(qnn.coefficients)

qnn.predict(samples, labels)
