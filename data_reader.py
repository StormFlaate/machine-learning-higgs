import numpy as np

def load_clean_csv(data_path, adjust_range=True):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    data = np.genfromtxt(data_path, delimiter=",")

    ids = data[:, 0].astype(np.int)
    y = data[:, 1].astype(np.int)
    tX = data[:, 2:-1]

    # if desired shift scope from [-1, 1] to [0, 1]
    if adjust_range:
        y[np.where(y == -1)] = 0

    return y, tX, ids


def load_weights(data_path):
    weights = np.genfromtxt(data_path, delimiter=",")
    return np.reshape(weights, (len(weights), 1))