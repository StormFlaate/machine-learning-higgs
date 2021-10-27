import numpy as np

#--------------- Writing Weights -----------------#
def weight_writer(w, DATA_SAVE_PATH):
    with open(DATA_SAVE_PATH, 'w') as f:
        f.write(str(w[0]))
        for weight in w[1:]:
            f.write("," + str(weight))

#--------------- Finding the MSE -----------------#
def compute_mse(y, tx, w):
    """Compute the MSE for given data and weights

    INPUT VARIABLES:
    - x                     Array of features
    - w                     Model weights for which the loss is to be calculated. Must have as many entries as x has columns
    - y                     1D array of data points. Must have as many rows as x

    OUTPUT VARIABLES:
    - mse                   The mean squared error (float)
    """

    e = y - tx.dot(w)
    mse = e.dot(e) / (len(e))
    return mse


#--------------- Building the k indices for cross validation -----------------#
def build_k_indices(y, k_fold, seed = 1):
    """ Function that builds the k-indices for k-fold cross validation

    Credit for writing this function goes to the TAs of the 2021-2022 ML course at EPFL

    INPUT VARIABLES:
    - y                     1D array of data points. Used to get the right dimensions for the indices array
    - k_fold                Number of groups into which the data is to be split for ridge regression (integer)
    - seed                  Integer or float to initialize the random number generator

    OUTPUT VARIABLES:
    - k_indices             Array containing shuffled indices. Has dimension k x int(len(y)/k)

    """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

#--------------- Split data for cross validation -----------------#
def split_data(x, y, k_indices, k):
    """
    WRITE DESCRIPTION
    """

    # Find number of columns of x
    #     -> necessary to account for special case of x being 1D
    if len(np.shape(x)) == 1:
        cols = 1
    else:
        cols = len(x[0])

    # Extract test set for y and x
    #      -> Reshaping is necessary in case that we are working with 1D arrays (i.e. for y and for some cases of x)
    y_test = np.reshape(y[k_indices[k]], (len(k_indices[0]), 1))
    x_test = np.reshape(x[k_indices[k]], (len(k_indices[0]), cols))


    # Extract remaining data
    #      -> flattening and reshaping is necessary to ensure the correct output shape
    if k == len(k_indices)-1:
        y_train = np.reshape(y[k_indices[:k]].flatten(), ((len(k_indices) - 1) * len(k_indices[0]), 1))
        x_train = np.reshape(x[k_indices[:k]].flatten(), ((len(k_indices) - 1) * len(k_indices[0]), cols))
    else:
        y_train = np.reshape(np.vstack((y[k_indices[:k]], y[k_indices[k+1:]])).flatten(), ((len(k_indices) - 1) * len(k_indices[0]), 1))
        x_train = np.reshape(np.vstack((x[k_indices[:k]], x[k_indices[k+1:]])).flatten(), ((len(k_indices) - 1) * len(k_indices[0]), cols))

    return x_train, y_train, x_test, y_test

#--------------- Cross Validation -----------------#
def cross_validation(y, x, k_fold, method, *args):
    """return the loss of ridge regression.

    WRITE DESCRIPTION

    """

    # Create shuffled indices for splitting the data into k groups
    k_indices = build_k_indices(y, k_fold)

    # Set up lists for training and test losses to be appended to for each iteration of the cross validation
    tr_mse_lst = []
    te_mse_lst = []

    # Loop over all different data splits
    for k in range(k_fold):

        # Split data into training and test sets
        x_train, y_train, x_test, y_test = split_data(x, y, k_indices, k)

        # Compute the weights using the specified method and any related args
        w = method(y_train, x_train, *args)

        # Compute the errors and append them to the respective lists
        #tr_mse_lst.append(compute_mse(y_train, x_train, w))
        te_mse_lst.append(compute_mse(y_test, x_test, w))

    # Finally take the averages over the collected lists

    tr_mse = sum(tr_mse_lst) / k_fold
    te_mse = sum(te_mse_lst) / k_fold

    return tr_mse, te_mse
