import torch
import argparse
import time
import os
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split, KFold
from algorithm.fact_qnn import FactorizedQNNClassical
from helpers import load_data, train_model, test_model

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser(description="Train Factorized QNN on datasets.")

    parser.add_argument("--dataset_dir", type=str, default="dataset/real_world", help="Dataset directory.")
    parser.add_argument("--d", type=str, default="all", help="Comma-separated list of dataset indexes.")
    parser.add_argument("--N", type=str, default="1,3,5,10", help="Comma-separated list of swap test counts.")
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
    parser.add_argument("--save_model", action=argparse.BooleanOptionalAction, default=True, help="Save the model file.")
    parser.add_argument("--seed", type=int, default=123, help="RNG seed.")
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=True, help="Verbose output.")

    args = parser.parse_args()

    datasets = sorted(os.listdir(args.dataset_dir))
    if args.d == 'all':
        data = datasets
    else:
        d_list = list(map(int, args.d.split(',')))
        data = [datasets[d] for d in d_list]

    N_list = list(map(int, args.N.split(',')))
    k_list = list(map(int, args.k.split(',')))
    lr_list = list(map(float, args.lr.split(',')))

    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)
    if not os.path.exists(args.modeldir):
        os.makedirs(args.modeldir)

    mytime = str(time.time())
    filename = f'results_{mytime}.csv'

    val_size = args.validation_size / 100

    res = []

    for dat in data:
        data_file = os.path.join(args.dataset_dir, dat)
        X, y = load_data(data_file, device)
        # KFold cross-validation
        kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
        for fold, (train_index, test_index) in enumerate(kf.split(X)):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            #validation
            if args.early_stopping:
                X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=args.seed)
            else:
                X_val, y_val = X_train, y_train
            d = X_train.shape[1]
            print(f"Training on dataset: {data_file}, fold: {fold}")
            for N in N_list:
                for k in k_list:
                    for lr in lr_list:
                        print(f"Training with N: {N}, k: {k}, lr: {lr}")
                        model = FactorizedQNNClassical(N, k, d).to(device)
                        train_model(model, X_train, y_train, X_val, y_val, compute_region=False, epochs=args.epochs, batch_size=args.batch_size, lr=lr, device=device, early_stopping=args.early_stopping, patience=args.patience, metric=args.validation_metric, verbose=args.verbose)
                        if args.save_model:
                            model_file = os.path.join(args.modeldir, f'model_{time.time()}.pt')
                            torch.save(model.state_dict(), model_file)
                        else:
                            model_file = None
                        y_hat = test_model(model, X_test, y_test)
                        accuracy = accuracy_score(y_test.cpu().numpy(), y_hat.cpu().numpy())
                        f1 = f1_score(y_test.cpu().numpy(), y_hat.cpu().numpy())
                        print(f'Accuracy: {accuracy}, F1: {f1}')
                        info = {'dataset': data_file, 'fold': fold, 'N': N, 'k': k, 'lr': lr, 'model_file': model_file}
                        args_dict = {f'args_{k}': v for k, v in vars(args).items()}
                        info = {**info, **args_dict}
                        res.append({**info, 'acc': accuracy, 'f1': f1})
                        pd.DataFrame(res).to_csv(os.path.join(args.logdir, filename), index=False)

            del X_train, y_train
            del X_test, y_test
            del X_val, y_val
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
