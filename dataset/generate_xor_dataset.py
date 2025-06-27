import pandas as pd
import numpy as np
import os
from data import generate_xor_balanced

np.random.seed(123)

dims = [1,2,3,4,5,6,7,8,9,10]
sizes = [100, 1000]
test_size = 0.2

for size in sizes:
    if not os.path.exists(f'synthetic/xor/train/{size}'):
        os.makedirs(f'synthetic/xor/train/{size}')
    if not os.path.exists(f'synthetic/xor/test/{size}'):
        os.makedirs(f'synthetic/xor/test/{size}')
    for dim in dims:
        header = ['x_{}'.format(i) for i in range(dim)] + ['class']
        # train
        X, y = generate_xor_balanced(dim, size)
        df = pd.DataFrame(np.concatenate([X, y.reshape(-1, 1)], axis=1))
        df.to_csv(f'synthetic/xor/train/{size}/xor_{dim}d.csv', index=False, header=header)
        # test
        X, y = generate_xor_balanced(dim, int(test_size * size))
        df = pd.DataFrame(np.concatenate([X, y.reshape(-1, 1)], axis=1))
        df.to_csv(f'synthetic/xor/test/{size}/xor_{dim}d.csv', index=False, header=header)
