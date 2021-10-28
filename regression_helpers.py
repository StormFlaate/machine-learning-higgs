import numpy as np

def build_poly(x, degree):
    """Creates polynomial basis functions for input data x, for j=0 up to j=degree.
    
    INPUT VARIABLES:
    - x                     Data array of size n x m where m can be equal to or greater than 1
    - verbose               Polynomial degree up to which the features shall be extended
    
    OUTPUT VARIABLES:
    - x_return              Array with the same number of rows as A but with d*m + 1 columns
    """

    # Reshape x in case that it is 1-dimensional
    if np.shape(x)[1] == 1:
        x = x.reshape(len(x), 1)
        
    # Add one column of 1s to x. Only one column is needed (not one per original feature)
    x_return = np.hstack((np.ones((len(x),1)), x))
    
    # Extend x by extending it by powers of the original features
    for i in range(2, degree + 1):
        x_return = np.hstack((x_return, x ** i))

    return x_return


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
        w, _ = method(y_train, x_train, *args)
        
        # Compute the errors and append them to the respective lists
        tr_mse_lst.append(compute_mse(y_train, x_train, w))
        te_mse_lst.append(compute_mse(y_test, x_test, w))
    
    # Finally take the averages over the collected lists

    tr_mse = sum(tr_mse_lst) / k_fold
    te_mse = sum(te_mse_lst) / k_fold

    return tr_mse, te_mse


def optimize_d_plus1(y, x, k_fold, method, d_range, param_range, *args):
    '''
    WRITE DESCRIPTION
    
    method must take arguments in the order y - x - parameter - other parameters
    '''
    
    # Create place for results to be stored
    res_tr = np.empty((len(d_range), len(param_range)))
    res_te = np.empty((len(d_range), len(param_range)))
    
    # Outer loop over the dimensions
    for d_count in range(len(d_range)):
        # define dimension
        d = d_range[d_count]
        
        # Build the data
        x = build_poly(x, d)
        
        # Inner loop over the instances of the parameter
        for p_count in range(len(param_range)):
            # define parameter value
            p = param_range[p_count]
            
            # compute losses for this combination
            loss_tr, loss_te = cross_validation(y, x, k_fold, method, p, *args)

            # store the results
            res_tr[d_count, p_count] = loss_tr
            res_te[d_count, p_count] = loss_te

    # Finding the minimum test error
    min_te_loss = np.amin(res_te)
    min_loss_coords = np.where(res_te == min_te_loss)

    # As one minimum is enough
    min_loss_coords = [min_loss_coords[0,0], min_loss_coords[1,0]]

    # Finding the corresponding best dimension and parameter
    d_best = d_range[min_loss_coords[0]]
    p_best = param_range[min_loss_coords[1]]

    # Finding the corresponding train error (in case that it is interesting)
    min_tr_loss = res_tr[min_loss_coords]

    # return the best dimension, best parameter, minimum losses, and collection of losses (might be nice for visualisation)
    return d_best, p_best, min_te_loss, min_tr_loss, res_te, res_tr

'''
Testing stuff


x = np.array([[1,2,3],
              [2,2,3],
              [3,2,3],
              [4,2,3],
              [5,2,3],
              [6,2,3]])

y = x[:,0]
k=2
k_indices = build_k_indices(y, 3, 1)

print(split_data(y, y, k_indices, 1))
print("bob")
print(split_data(x, y, k_indices, 2))






'''