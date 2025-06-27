import torch
import argparse
import time
import os
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split, KFold
from algorithm.fact_qnn import FactorizedQNNClassical_split
from helpers import  train_model, test_model
from algorithm.helpers import n_choose_k
from sklearn.datasets import fetch_openml
import numpy as np

def reorder_by_quadrants(X, n_divisions=4):
    n_samples = X.shape[0]
    img_size = 28
    n_per_side = int(np.sqrt(n_divisions))
    if n_per_side ** 2 != n_divisions:
        raise ValueError(f"n_divisions must be a perfect square, got {n_divisions}")
    div_height = img_size // n_per_side
    div_width = img_size // n_per_side
    used_rows = div_height * n_per_side
    used_cols = div_width * n_per_side
    if used_rows < img_size or used_cols < img_size:
        ignored_rows = img_size - used_rows
        ignored_cols = img_size - used_cols
        print(f"WARNING: Image cannot be evenly divided into {n_divisions} equal quadrants.")
        print(f"Ignoring {ignored_rows} rows and {ignored_cols} columns.")
        print(f"Using {used_rows}x{used_cols} pixels out of {img_size}x{img_size}.")
    new_features = used_rows * used_cols
    X_reordered = np.zeros((n_samples, new_features))
    for sample_idx in range(n_samples):
        img_2d = X[sample_idx].reshape(img_size, img_size)
        new_flat = np.zeros(new_features)
        flat_idx = 0
        for row_div in range(n_per_side):
            for col_div in range(n_per_side):
                row_start = row_div * div_height
                row_end = (row_div + 1) * div_height
                col_start = col_div * div_width
                col_end = (col_div + 1) * div_width
                division = img_2d[row_start:row_end, col_start:col_end]
                division_flat = division.flatten()
                new_flat[flat_idx:flat_idx + len(division_flat)] = division_flat
                flat_idx += len(division_flat)
        X_reordered[sample_idx] = new_flat
    return X_reordered

def load_mnist_binary_classifier(digit1, digit2, split=False, split_ratio=0.8, quadrant_reorder=None, subset_fraction=1.0, seed=None):
    print("Downloading MNIST dataset (this may take a moment)...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='liac-arff')
    X = mnist.data.astype('float32') / 255.0
    y = mnist.target.astype('int')
    mask = (y == digit1) | (y == digit2)
    X_binary = X[mask]
    y_binary = y[mask]
    y_binary = np.where(y_binary == digit1, 1, -1)
    if quadrant_reorder is not None:
        X_binary = reorder_by_quadrants(X_binary, quadrant_reorder)
    if 0 < subset_fraction < 1.0:
        n_total = X_binary.shape[0]
        n_subset = int(n_total * subset_fraction)
        rng = np.random.default_rng(seed)
        indices = rng.choice(n_total, n_subset, replace=False)
        X_binary = X_binary[indices]
        y_binary = y_binary[indices]
    if not split:
        return X_binary, y_binary
    n_samples = X_binary.shape[0]
    shuffle_idx = np.random.permutation(n_samples)
    n_train = int(n_samples * split_ratio)
    X_train = X_binary[shuffle_idx[:n_train]]
    y_train = y_binary[shuffle_idx[:n_train]]
    X_test = X_binary[shuffle_idx[n_train:]]
    y_test = y_binary[shuffle_idx[n_train:]]
    return X_train, y_train, X_test, y_test


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser(description="Train Factorized QNN on MNIST for binary digit classification tasks, inout split onto multiple modules.")

    
    parser.add_argument("--digit_pairs", type=str, default="0-1", help="Comma-separated list of digit pairs (e.g., 0-1,3-5,7-9), uses all 45 possible pairs if set to 'all'")
    parser.add_argument("--subset_fraction", type=float, default=1.0, help="Fraction of data to use from the full dataset (0 < f <= 1.0).")
    parser.add_argument("--quadrants", type=int, default=4, help="Images are reordered into quadrants. Must be perfect square.")
    parser.add_argument("--Nsplit", type=str, default="1,2", help="Comma-separated list of swap test counts for each partition of the image (overall quadrants * Nsplit swap tests).")
    
    parser.add_argument("--k", type=str, default="1,2,3", help="Comma-separated list of factor module counts.")
    parser.add_argument("--lr", type=str, default="1", help="Comma-separated list of learning rates.")
    parser.add_argument("--epochs", type=int, default=50000, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=256000, help="Batch size for training and testing data.")
    parser.add_argument("--early_stopping", action=argparse.BooleanOptionalAction, default=True, help="Early stopping.")
    parser.add_argument("--validation_size", type=int, default=20, help="Validation set size percentage.")
    parser.add_argument("--n_splits", type=int, default=10, help="Number of cross-validation folds.")
    parser.add_argument("--patience", type=int, default=5000, help="Patience for early stopping.")
    parser.add_argument("--validation_metric", type=str, default='accuracy', help="Metric chosen for validation.")
    parser.add_argument("--logdir", type=str, default='logs/data_kfold', help="Logging directory.")
    parser.add_argument("--modeldir", type=str, default='models/data_kfold', help="Models directory.")
    parser.add_argument("--save_model", action=argparse.BooleanOptionalAction, default=False, help="Save the model file.")
    parser.add_argument("--seed", type=int, default=123, help="RNG seed.")
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=False, help="Verbose output.")
  

    args = parser.parse_args()

    Nsplit_list = list(map(int, args.Nsplit.split(',')))
    k_list = list(map(int, args.k.split(',')))
    lr_list = list(map(float, args.lr.split(',')))
    val_size = args.validation_size / 100

    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)
    if not os.path.exists(args.modeldir):
        os.makedirs(args.modeldir)

    mytime = str(time.time())
    filename = f'results_{mytime}.csv'
    res = []
    
    if args.digit_pairs == 'all':
        digit_pairs = [tuple(map(int, pair.split('-'))) for pair in n_choose_k(10, 2).split(',')]
    else:
        digit_pairs = [tuple(map(int, pair.split('-'))) for pair in args.digit_pairs.split(',')]
        
    

    for digit1, digit2 in digit_pairs:
        dataset_name = f"MNIST_{digit1}_vs_{digit2}"
        X_np, y_np = load_mnist_binary_classifier(digit1, digit2, split=False,quadrant_reorder=args.quadrants, subset_fraction=args.subset_fraction, seed=args.seed)
        X = torch.tensor(X_np, dtype=torch.float32).to(device)
        y = torch.tensor(y_np, dtype=torch.long).to(device)
        
        kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)

        for fold, (train_index, test_index) in enumerate(kf.split(X)):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            if args.early_stopping:
                X_train, X_val, y_train, y_val = train_test_split(
                    X_train, y_train, test_size=val_size, random_state=args.seed
                )
            else:
                X_val, y_val = X_train, y_train
                
            d = X_train.shape[1]
            print(f"Training on {dataset_name}, fold: {fold}")
            for Nsplit in Nsplit_list:
                for k in k_list:
                    for lr in lr_list:
                        print(f"Training with Splits: {args.quadrants}, Nsplit: {Nsplit}, k: {k}, lr: {lr}")
                        model = FactorizedQNNClassical_split(Nsplit, k, d, args.quadrants).to(device)
                        train_model(model, X_train, y_train, X_val, y_val,
                                    compute_region=False, epochs=args.epochs, batch_size=args.batch_size, lr=lr,
                                    device=device, early_stopping=args.early_stopping, patience=args.patience,
                                    metric=args.validation_metric, verbose=args.verbose)

                        model_file = None
                        if args.save_model:
                            model_file = os.path.join(args.modeldir, f'model_{dataset_name}_fold{fold}_{time.time()}.pt')
                            torch.save(model.state_dict(), model_file)

                        y_hat = test_model(model, X_test, y_test)
                        accuracy = accuracy_score(y_test.cpu().numpy(), y_hat.cpu().numpy())
                        f1 = f1_score(y_test.cpu().numpy(), y_hat.cpu().numpy())
                        print(f'Accuracy: {accuracy}, F1: {f1}')

                        info = {'dataset': dataset_name, 'fold': fold, 'Splits': args.quadrants, 'Nsplit': Nsplit, 'k': k, 'lr': lr, 'model_file': model_file}
                        args_dict = {f'args_{k}': v for k, v in vars(args).items()}
                        res.append({**info, **args_dict, 'acc': accuracy, 'f1': f1})
                        pd.DataFrame(res).to_csv(os.path.join(args.logdir, filename), index=False)

            del X_train, y_train, X_test, y_test, X_val, y_val
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
