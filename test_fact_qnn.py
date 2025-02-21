import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from algorithm.fact_qnn import FactorizedQNNClassical
from dataset.data import generate_xor

def train_model(model, X_train, y_train, X_test, y_test, epochs=100, lr=0.01, batch_size=32):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()  # sigmoid and log loss

    for epoch in range(epochs):
        model.train()
        permutation = np.random.permutation(X_train.size(0))  # permute batch

        for i in np.arange(0, X_train.size(0), batch_size):
            indices = permutation[i:i+batch_size]
            x_batch, y_batch = X_train[indices], y_train[indices]

            # forward pass
            y_hat = model(x_batch).squeeze()
            loss = criterion(y_hat, (y_batch + 1) // 2)

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # evaluation
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                z = model(X_test).squeeze()
                y_pred = torch.sign(z)
                accuracy = (y_pred == y_test).float().mean()
            print(f"epoch {epoch+1}, \t loss: {loss.item():.4f}, \t accuracy: {accuracy:.2f}")

N = 10  # number of swap tests
k = 2  # number of factor modules
d = 3  # number of features (xor dim)

num_samples = 1000 

X, y = generate_xor(d, num_samples)  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

X_train, y_train = torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)
X_test, y_test = torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)

model = FactorizedQNNClassical(N, k, d)

train_model(model, X_train, y_train, X_test, y_test, epochs=1000, lr=0.01)
