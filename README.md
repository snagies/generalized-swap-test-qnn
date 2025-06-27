# Enhancing Expressivity of Quantum Neural Networks Based on the SWAP test
Code for the implementation of "Enhancing Expressivity of Quantum Neural Networks Based on the SWAP test"

## Project Structure
- `algorithm/`: Core implementation of the SWAP test-based QNN and utility functions.
- `dataset/`: Contains datasets used for training and testing and the data loading functions.
- `test_datasets_kfold.py`, `test_qnn_datasets_kfold.py`: Scripts to run k-fold tests on datasets.
- `test_xor_classical.py`, `test_xor_quantum.py`: Scripts to run the XOR test.
- `test_kfold_MNIST.py`, `test_kfold_MNIST_split.py`: Scripts for MNIST tests.
- `requirements.txt`: Python dependencies.
- `README.md`: Project documentation.
- `LICENSE`: License file.

## Setup
### Get the code
    git clone https://github.com/snagies/generalized-swap-test-qnn.git
    cd generalized-swap-test-qnn

### Install dependencies
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt

## Run
### Run the tests on real world datasets
    python test_datasets_kfold.py --dataset_dir dataset/real_world

### Run the XOR test
#### Generate the dataset
    cd dataset
    python generate_xor_dataset.py
    cd ..
#### Run the test
    python test_xor_quantum.py

### Run the MNIST test
    python test_kfold_MNIST.py
    python test_kfold_MNIST_split.py

### Run the spirals test
#### Generate the dataset
    cd dataset
    python generate_spirals_dataset.py
    cd ..
#### Run the test
    python test_datasets_kfold.py --dataset_dir dataset/synthetic/spirals

### Options

- `--dataset_dir`: Directory containing the dataset to be used for training/testing (default: `dataset/real_world`).
- `--d`: Comma-separated list of dataset indexes or feature dimensions, or `all` for all datasets (default: `all`).
- `--N`: Comma-separated list of swap test counts (default: `1,3,5,10`).
- `--k`: Comma-separated list of factor module counts (default: `1,2,3`).
- `--lr`: Comma-separated list of learning rates (default: `1`).
- `--epochs`: Number of training epochs (default: `50000`).
- `--batch_size`: Batch size for training and testing data (default: `256000`).
- `--early_stopping`: Enable early stopping (default: `True`).
- `--validation_size`: Validation set size percentage (default: `20`).
- `--n_splits`: Number of splits for k-fold cross-validation (default: `10`).
- `--patience`: Patience for early stopping (default: `5000`).
- `--validation_metric`: Metric chosen for validation (default: `accuracy`).
- `--logdir`: Directory where the results/logs will be saved (default: `logs/data_kfold`).
- `--modeldir`: Directory where the models will be saved (default: `models/data_kfold`).
- `--save_model`: Save the trained model file (default: `True`).
- `--seed`: RNG seed for reproducibility (default: `123`).
- `--verbose`: Verbose output (default: `True`).
- `--shots`: Number of shots for quantum execution (default: `8192`).
- `--s`: Comma-separated list of dataset sizes (default: `1000` for synthetic tests).
- `--Nsplit`: Comma-separated list of swap test counts for each partition of the image (default: `1,2`).
- `--digit_pairs`: Comma-separated list of digit pairs for classification (e.g., `0-1,3-5`), or `all` for all pairs (default: `0-1`).
- `--subset_fraction`: Fraction of data to use from the full dataset (default: `1.0`).
- `--quadrants`: Number of quadrants to split images into (default: `4`).

*Note: Some options are specific to certain test scripts (e.g., `--digit_pairs`, `--quadrants`, `--shots`). See the relevant script for details.*