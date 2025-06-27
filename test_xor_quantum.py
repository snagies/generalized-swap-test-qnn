import torch
import os
import argparse
import time
import pandas as pd
from sklearn.metrics import accuracy_score
from algorithm.fact_qnn_quantum import FactorizedQnnQuantum
from helpers import load_data

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser(description="Train Factorized QNN on XOR synthetic datasets.")

    parser.add_argument("--d", type=str, default="3", help="Comma-separated list of feature dimensions.")
    parser.add_argument("--s", type=str, default="100", help="Comma-separated list of dataset sizes.")
    parser.add_argument("--N", type=str, default="4", help="Comma-separated list of swap test counts.")
    parser.add_argument("--k", type=str, default="2", help="Comma-separated list of factor module counts.")
    parser.add_argument("--lr", type=str, default="1", help="Comma-separated list of learning rates.")
    parser.add_argument("--epochs", type=int, default=50000, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=256000, help="Batch size for training and testing data.")
    parser.add_argument("--shots", type=int, default="8192", help="Number of shots for quantum execution.")
    parser.add_argument("--logdir", type=str, default='logs/xor_quantum', help="Logging directory.")
    parser.add_argument("--modeldir", type=str, default='models/xor_quantum', help="Models directory.")
    parser.add_argument("--save_model", action=argparse.BooleanOptionalAction, default=True, help="Save the model file.")
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=True, help="Verbose output.")

    args = parser.parse_args()

    d_list = list(map(int, args.d.split(',')))
    s_list = list(map(int, args.s.split(',')))
    N_list = list(map(int, args.N.split(',')))
    k_list = list(map(int, args.k.split(',')))
    lr_list = list(map(float, args.lr.split(',')))

    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)
    if not os.path.exists(args.modeldir):
        os.makedirs(args.modeldir)

    mytime = str(time.time())

    filename = f'XOR_quantum_{mytime}.csv'

    res = []
    shots = args.shots

    for d in d_list:
        for s in s_list:
            train_file = f'dataset/synthetic/xor/train/{s}/xor_{d}d.csv'
            test_file = f'dataset/synthetic/xor/test/{s}/xor_{d}d.csv'

            X_train, y_train = load_data(train_file, device)
            X_test, y_test = load_data(test_file, device)

            for N in N_list:
                for k in k_list:
                    for lr in lr_list:
                        model = FactorizedQnnQuantum(N, k, d)
                        model.train_classical(X_train, y_train, X_test, y_test, device, epochs=args.epochs, batch_size=args.batch_size, lr=lr, verbose=args.verbose)
                        if args.save_model:
                            model_file = os.path.join(args.modeldir, f'model_{time.time()}.pt')
                            torch.save(model.classical_model.state_dict(), model_file)
                        else:
                            model_file = None
                        info = {'d': d, 's': s, 'N': N, 'k': k, 'lr': lr, 'shots':shots, 'epochs': args.epochs, 'batch_size': args.batch_size, 'model_file': model_file}
                        args_dict = {f'args_{k}': v for k, v in vars(args).items()}
                        info = {**info, **args_dict}

                        y_hat = model.predict_classical(X_test)
                        acc = accuracy_score(y_test.cpu().numpy(), y_hat.cpu().numpy())
                        res.append({'runner': 'classical', **info, 'acc': acc})
                        print(f'Accuracy classical: {acc}')

                        y_hat = model.predict_quantum(X_test, execution='statevector')
                        acc = accuracy_score(y_test.cpu().numpy(), y_hat.cpu().numpy())
                        res.append({'runner': 'statevector', **info, 'acc': acc})
                        print(f'Accuracy statevector: {acc}')   

                        y_hat = model.predict_quantum(X_test, execution='simulator', shots=shots)
                        acc = accuracy_score(y_test.cpu().numpy(), y_hat.cpu().numpy())
                        res.append({'runner': 'simulator', **info, 'acc': acc})
                        print(f'Accuracy simulator: {acc}')

                        # y_hat = model.predict_quantum(X_test, execution='real', shots=shots)
                        # acc = accuracy_score(y_test.cpu().numpy(), y_hat.cpu().numpy())
                        # res.append({'runner': 'real', **info, 'acc': acc})
                        # print(f'Accuracy real: {acc}')

                        pd.DataFrame(res).to_csv(os.path.join(args.logdir, filename), index=False)

            del X_train, y_train
            del X_test, y_test
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
