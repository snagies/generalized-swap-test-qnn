import pandas as pd
import numpy as np
import os
from data import generate_spirals

np.random.seed(123)

if not os.path.exists('synthetic/spirals'):
    os.makedirs('synthetic/spirals')

rounds = [1, 2, 3]

for r in rounds:
    X, y = generate_spirals(1000, 0.04, n_rounds=r, shuffle=True)
    pd.DataFrame(np.concatenate([X, np.linalg.norm(X, axis=1).reshape(-1, 1), y.reshape(-1, 1)], axis=1), columns=['x1', 'x2', 'norm', 'y']).to_csv(f'synthetic/spirals/spirals_n{r}.csv', index=False)
