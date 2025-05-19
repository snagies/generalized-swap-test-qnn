import torch
import torch.optim as optim
import torch.nn as nn
import pandas as pd
from sklearn.metrics import f1_score

def load_data(filename, device):
    df = pd.read_csv(filename)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return torch.tensor(X, dtype=torch.float32).to(device), torch.tensor(y, dtype=torch.float32).to(device)

def train_model(model, X_train, y_train, X_test, y_test, device, epochs=1000, batch_size=100000, lr=0.01, writer=None, verbose=True, compute_region=True, early_stopping=False, check_every=10, patience=100, metric='accuracy'):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    if compute_region:
        region_ids, region_counts = compute_regions(X_test, device)

    if early_stopping:
        best_perf = 0
        patience_counter = 0

    for epoch in range(epochs):
        model.train()

        indices = torch.randperm(X_train.size(0), device=device)  # Generate shuffled indices on the same device
        X_train = X_train[indices]
        y_train = y_train[indices]

        for i in range(0, X_train.size(0), batch_size):
            X_batch = X_train[i : i + batch_size]
            y_batch = y_train[i : i + batch_size]

            optimizer.zero_grad()

            y_hat = model(X_batch).squeeze()
            loss = criterion(y_hat, (y_batch + 1) / 2)

            loss.backward()
            optimizer.step()

        if (epoch + 1) % check_every == 0:
            y_pred = test_model(model, X_test, y_test, verbose=False)
            predictions = (y_pred == y_test).float()
            if metric == 'accuracy':
                perf = predictions.mean().detach().cpu().item()
            elif metric == 'f1':
                perf = f1_score(y_test.cpu().numpy(), y_pred.cpu().numpy())
            else:
                raise ValueError("Invalid metric. Choose 'accuracy' or 'f1'.")
            if verbose:
                print(f"epoch {epoch+1}, \t loss: {loss.item():.4f}, \t {metric}: {perf:.2f}")
            if writer is not None:
                writer.add_scalar(f"Train/Loss", loss.item(), epoch)
                writer.add_scalar(f"Test/{metric.capitalize()}", perf, epoch)
                if compute_region:
                    region_accuracy = compute_region_accuracy(predictions, region_ids, region_counts)
                    writer.add_histogram("Test/RegionAccuracy", region_accuracy, epoch)
                    #for region_id, acc in enumerate(region_accuracy):
                        #writer.add_scalar(f"Test/Region_{region_id}_Accuracy", acc.item(), epoch)
            if perf == 1:
                if verbose:
                    print('Early stopping.')
                return False
            if early_stopping:
                # implement early stopping based on last performance
                if perf > best_perf:
                    best_perf = perf
                    patience_counter = 0
                else:
                    patience_counter += check_every
                    if patience_counter >= patience:
                        if verbose:
                            print('Early stopping.')
                        return False
    return True

def test_model(model, X_test, y_test, verbose=False):
    model.eval()
    with torch.no_grad():
        z = model(X_test).squeeze()
        y_pred = torch.sign(z)
        y_pred[y_pred == 0] = -1
    if verbose:
        accuracy = (y_pred == y_test).float().mean().detach().cpu().item()
        print(f"accuracy: {accuracy:.2f}")
    return y_pred

def compute_regions(X, device):
    bin_vec = (X > 0).int()
    region_ids = torch.sum(bin_vec * (2 ** torch.arange(X.shape[1], device=device)), dim=1)
    region_counts = torch.bincount(region_ids, minlength=region_ids.max() + 1)
    return region_ids, region_counts

def compute_region_accuracy(predictions, region_ids, region_counts):
    region_correct = torch.bincount(region_ids, weights=predictions, minlength=region_ids.max() + 1)
    region_accuracy = region_correct / region_counts
    return region_accuracy
