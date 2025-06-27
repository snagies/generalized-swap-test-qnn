import torch
import argparse
from torch.utils.tensorboard import SummaryWriter
from algorithm.fact_qnn import FactorizedQNNClassical
from helpers import load_data, train_model

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser(description="Train Factorized QNN on XOR synthetic datasets.")

    parser.add_argument("--d", type=str, default="1,2,3,4,5,6,7,8,9,10", help="Comma-separated list of feature dimensions.")
    parser.add_argument("--s", type=str, default="1000", help="Comma-separated list of dataset sizes.")
    parser.add_argument("--N", type=str, default="10,100,1000", help="Comma-separated list of swap test counts.")
    parser.add_argument("--k", type=str, default="1,2,3,4,5,6,7", help="Comma-separated list of factor module counts.")
    parser.add_argument("--lr", type=str, default="0.01,0.1,1,10", help="Comma-separated list of learning rates.")
    parser.add_argument("--epochs", type=int, default=50000, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=256000, help="Batch size for training and testing data.")
    parser.add_argument("--logdir", type=str, default='logs/xor_classical', help="Logging directory.")

    args = parser.parse_args()

    d_list = list(map(int, args.d.split(',')))
    s_list = list(map(int, args.s.split(',')))
    N_list = list(map(int, args.N.split(',')))
    k_list = list(map(int, args.k.split(',')))
    lr_list = list(map(float, args.lr.split(',')))

    for d in d_list:
        for s in s_list:
            train_file = f'dataset/synthetic/xor/train/{s}/xor_{d}d.csv'
            test_file = f'dataset/synthetic/xor/test/{s}/xor_{d}d.csv'

            X_train, y_train = load_data(train_file, device)
            X_test, y_test = load_data(test_file, device)

            for N in N_list:
                for k in k_list:
                    for lr in lr_list:
                        writer = SummaryWriter(f"{args.logdir}/XOR_test_d:{d}_s:{s}_N:{N}_k:{k}_lr:{lr}")

                        model = FactorizedQNNClassical(N, k, d).to(device)
                        cont = train_model(model, X_train, y_train, X_test, y_test, device, epochs=args.epochs, batch_size=args.batch_size, lr=lr, writer=writer)

                        writer.close()

                        if not cont:
                            break

            del X_train, y_train
            del X_test, y_test
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
