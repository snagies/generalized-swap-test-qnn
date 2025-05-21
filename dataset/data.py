import numpy as np
import os
import pathlib
import random
from sklearn.datasets import fetch_openml

#01_iris_setosa_versicolor 01_iris_versicolor_virginica
def get_iris(dataset = '01_iris_setosa_versicolor.csv', permute = True, size = 'all'):
    #TODO cleanup; make portable
    folderpath = pathlib.Path(__file__).parent.resolve()
    folderpath = folderpath.__str__()
    file = os.path.join(folderpath, dataset)
    
    data = np.genfromtxt(file, delimiter = ',', skip_header=1)
    
    if permute == True:
        data = np.random.permutation(data)
    
    if size == 'all':
        samples = data[:,:-1]
        labels = data[:,-1]
    #TODO ensure equal frequency in classes if size != 'all'  
    else:
        samples = data[:size,:-1]
        labels = data[:size,-1]
        
    return samples, labels

def generate_xor_old(dim, size):
    samples = []
    labels = []
    counter = 0
    for i in range(size):
        r = random.randint(0, 2**dim-1) #random.randint(0, 2**dim)
        x = []
        temp = r
        y = 0
        for d in range(dim):
            s = int((temp % 2))
            x.append((1 if s == 0 else -1) * random.random())
            y += s
            temp //= 2
        y = 1 if (y % 2) == 0 else -1
        if y == 1:
            counter += 1
        samples.append(x)
        labels.append(y)
    print(counter, ' instances with label +1')
    return samples, labels


def generate_xor(dim, size):
    #new version
    samples = 2 * np.random.random(size=(size, dim)) - 1
    labels = np.sign(np.prod(samples, axis=1))
    counter = np.sum(labels == 1)
    print(counter, ' instances with label +1')
    return samples, labels

def reorder_by_quadrants(X, n_divisions=4):
    """
    Reorder the flattened image vectors by quadrants of equal size.
    If the image cannot be evenly divided, remaining rows/columns will be ignored.
    
    Parameters:
    X (numpy.ndarray): Input data matrix, each row is a flattened 28x28 image (784 pixels)
    n_divisions (int): Number of divisions (must be a perfect square: 4, 9, 16, etc.)
    
    Returns:
    numpy.ndarray: Reordered data matrix with possibly reduced features if pixels are ignored
    """
    n_samples = X.shape[0]
    img_size = 28  # MNIST images are 28x28
    
    # Check if n_divisions is a perfect square
    n_per_side = int(np.sqrt(n_divisions))
    if n_per_side ** 2 != n_divisions:
        raise ValueError(f"n_divisions must be a perfect square, got {n_divisions}")
    
    # Calculate the size of each division (must be equal)
    div_height = img_size // n_per_side
    div_width = img_size // n_per_side
    
    # Calculate how many rows/columns will be used
    used_rows = div_height * n_per_side
    used_cols = div_width * n_per_side
    
    # Check if we need to ignore some rows/columns
    if used_rows < img_size or used_cols < img_size:
        ignored_rows = img_size - used_rows
        ignored_cols = img_size - used_cols
        print(f"WARNING: Image cannot be evenly divided into {n_divisions} equal quadrants.")
        print(f"Ignoring {ignored_rows} rows and {ignored_cols} columns.")
        print(f"Using {used_rows}x{used_cols} pixels out of {img_size}x{img_size}.")
    
    # Calculate number of features in the new vector
    new_features = used_rows * used_cols
    
    # Initialize output array with potentially fewer features
    X_reordered = np.zeros((n_samples, new_features))
    
    for sample_idx in range(n_samples):
        # Reshape to 2D
        img_2d = X[sample_idx].reshape(img_size, img_size)
        
        # Create a new flattened array
        new_flat = np.zeros(new_features)
        flat_idx = 0
        
        # Go through each division in order (top-left, top-right, bottom-left, bottom-right, etc.)
        for row_div in range(n_per_side):
            for col_div in range(n_per_side):
                # Calculate the division boundaries
                row_start = row_div * div_height
                row_end = (row_div + 1) * div_height
                col_start = col_div * div_width
                col_end = (col_div + 1) * div_width
                
                # Extract the division and flatten it
                division = img_2d[row_start:row_end, col_start:col_end]
                division_flat = division.flatten()
                
                # Add to the new flattened array
                new_flat[flat_idx:flat_idx + len(division_flat)] = division_flat
                flat_idx += len(division_flat)
        
        X_reordered[sample_idx] = new_flat
    
    return X_reordered

def load_mnist_binary_classifier(digit1, digit2, split=False, split_ratio=0.8, quadrant_reorder=None):
    """
    Load MNIST dataset for binary classification between two specified digits.
    
    Parameters:
    digit1, digit2 (int): The two digits to classify between (0-9)
    split (bool): Whether to split into train/test sets (default: False)
    split_ratio (float): Ratio of data to use for training if split=True (default: 0.8)
    quadrant_reorder (int or None): If set, reorganizes pixels by image quadrants
                                    For example, if set to 4, pixels are grouped by
                                    top-left, top-right, bottom-left, bottom-right quadrants
                                    If image can't be evenly divided, some pixels will be ignored
    
    Returns:
    If split=False:
        X_binary (numpy.ndarray): Flattened images (one-dimensional vectors)
        y_binary (numpy.ndarray): Labels as +1 (for digit1) and -1 (for digit2)
    If split=True:
        X_train, y_train, X_test, y_test: Split datasets and labels
    """
    print("Downloading MNIST dataset (this may take a moment)...")
    
    # Download the dataset with parameters to avoid pandas dependency
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='liac-arff')
    
    # Get data and target
    X = mnist.data.astype('float32') / 255.0  # Normalize to [0, 1]
    y = mnist.target.astype('int')  # Convert string labels to integers
    
    # Filter to keep only the two digits we want
    mask = (y == digit1) | (y == digit2)
    X_binary = X[mask]
    y_binary = y[mask]
    
    # Convert labels to +1 and -1
    y_binary = np.where(y_binary == digit1, 1, -1)
    
    # Apply quadrant reordering if specified
    if quadrant_reorder is not None:
        X_binary = reorder_by_quadrants(X_binary, quadrant_reorder)
    
    # Return full dataset if split is False
    if not split:
        print(f"Loaded {X_binary.shape[0]} samples for digits {digit1} and {digit2}")
        print(f"Each image is represented as a vector of {X_binary.shape[1]} features")
        return X_binary, y_binary
    
    # Otherwise, split into train and test sets
    n_samples = X_binary.shape[0]
    shuffle_idx = np.random.permutation(n_samples)
    
    n_train = int(n_samples * split_ratio)
    X_train = X_binary[shuffle_idx[:n_train]]
    y_train = y_binary[shuffle_idx[:n_train]]
    X_test = X_binary[shuffle_idx[n_train:]]
    y_test = y_binary[shuffle_idx[n_train:]]
    
    print(f"Loaded {n_samples} samples for digits {digit1} and {digit2}")
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Each image is represented as a vector of {X_train.shape[1]} features")
    
    return X_train, y_train, X_test, y_test






# Example: Load digits 1 and 0
#X_train, Y_train = load_mnist_binary_classifier(1, 0, quadrant_reorder=4)

# Reshape one image back to 28x28 to visualize 
# import matplotlib.pyplot as plt
# plt.figure(figsize=(3, 3))
# plt.imshow(X_train[600,1*196:2*196].reshape(14, 14), cmap='gray')
# plt.title(f"Label: {'+1' if Y_train[600] > 0 else '-1'}")
# plt.axis('off')
# plt.show()



