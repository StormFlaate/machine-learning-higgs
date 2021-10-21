# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import numpy as np


def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(y.shape[0])
    yb[np.where(y=='b')] = -1
    
    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids


def predict_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1
    
    return y_pred


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in .csv format for submission to Kaggle or AIcrowd
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


# ************************************************************************************************
# Helper functions for implementation.py *********************************************************
# ************************************************************************************************

def sigmoid(t):
    """Sigmoid function
    
    INPUT VARIABLES:
    - t:        Given variable

    OUTPUT VARIABLES:
    - sigmoid:  The value of sigmoid function given variable t
    """

    sigmoid = 1/(1+np.exp(-t))
    return sigmoid


def compute_mean_square_error(y, tx, w):
    """Calculate the mean square error.
    
    INPUT VARIABLES:
    - y:     Observed data (Vector: Nx1)
    - tx:    Input data (Matrix: NxD) 
    - w:     Weigths (Vector: Dx1)
    
    LOCAL VARIABLES:
    - N:     Number of datapoints
    - e:     Error (Vector: Nx1)

    OUPUT VARIABLES:
    - mse:   Mean square error (Scalar)
    """
    N = y.shape[0]
    # Loss by MSE (Mean Square Error)
    e =  y - tx@w
    mse = (1/(2*N))*e.T@e
    return mse 


def compute_least_squares_gradient(y, tx, w):
    """Compute the gradient.
    
    INPUT VARIABLES:
    - y:     Observed data (Vector: Nx1)
    - tx:    Input data (Matrix: NxD) 
    - w:     Weigths (Vector: Dx1)
    
    LOCAL VARIABLES:
    - N:     Number of datapoints
    - e:     Error (Vector: Nx1)

    OUPUT VARIABLES:
    - gradient:    Gradient (Vector: Dx1)
    """
    
    N = y.shape[0]    
    e = y-tx@w
    gradient = -(1/N)*tx.T@e
    return gradient


def compute_least_squares_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient
    
    INPUT VARIABLES:
    - y:           Observed data (Vector: Nx1)
    - tx:          Input data (Matrix: NxD) 
    - w:           Weights (Vector: Dx1)
    
    OUTPUT VARIABLES:
    - gradient:    Gradient (Vector: Dx1)
    """
    # TODO: implement stochastic gradient computation.It's same as the gradient descent.
    N = y.shape[0]    
    e = y-tx@w
    gradient = -(1/N)*tx.T@e
    return gradient


def compute_negative_log_likelihood_loss(y, tx, w):
    """Compute a stochastic gradient
    
    INPUT VARIABLES:
    - y:           Observed data (Vector: Nx1)
    - tx:          Input data (Matrix: NxD) 
    - w:           Weights (Vector: Dx1)
    
    OUTPUT VARIABLES:
    - gradient:    Gradient (Vector: Dx1)
    """

    N = y.shape[0]
    tot_sum = 0
    for i, y_i in enumerate(y):
        tot_sum += np.log(1+np.exp(tx[i].T@w)) - y_i*tx[i].T@w
    return np.sum(tot_sum)


def compute_negative_log_likelihood_gradient(y, tx, w):
    """Compute a negative log likelihood gradient
    
    INPUT VARIABLES:
    - y:           Observed data (Vector: Nx1)
    - tx:          Input data (Matrix: NxD) 
    - w:           Weights (Vector: Dx1)
    
    OUTPUT VARIABLES:
    - gradient:    Gradient (Vector: Dx1)
    """
    gradient = tx.T@(sigmoid(tx@w)-y)
    return gradient


def logistic_regression_gradient_descent_one_step(y, tx, w, gamma):
    """Do one step of logistic regression with gradient descent
    
    INPUT VARIABLES:
    - y:           Observed data (Vector: Nx1)
    - tx:          Input data (Matrix: NxD) 
    - w:           Weights (Vector: Dx1)
    - gamma:       Step size for the stoch gradient descent (Scalar/constant)
    
    OUTPUT VARIABLES:
    - loss:        Mean square error of weights w (Scalar)
    - w:           Weights calculated (Vector: Dx1)
    """

    loss = compute_negative_log_likelihood_loss(y,tx,w)
    gradient = compute_negative_log_likelihood_gradient(y,tx,w)
    
    # Updating the w
    w = w - gamma*gradient
    
    return w, loss


def logistic_regression_stoch_gradient_descent_one_step(y, tx, w, batch_size, gamma):
    """Do one step of logistic regression with stoch gradient descent
    
    INPUT VARIABLES:
    - y:           Observed data (Vector: Nx1)
    - tx:          Input data (Matrix: NxD) 
    - w:           Weights (Vector: Dx1)
    - batch_size:  Number of elements that will be used per iteration for the stoch gradient descent
    - gamma:       Step size for the stoch gradient descent (Scalar/constant)
    
    OUTPUT VARIABLES:
    - loss:        Mean square error of weights w (Scalar)
    - w:           Weights calculated (Vector: Dx1)
    """

    for y_n, tx_n in batch_iter(y,tx, batch_size):
        gradient = compute_negative_log_likelihood_gradient(y_n,tx_n,w)
        # Updating the w
        w = w - gamma*gradient
    
    loss = compute_negative_log_likelihood_loss(y,tx,w)

    return w, loss


def penalized_logistic_regression_gradient_descent_one_step(y, tx, w, gamma, lambda_):
    """Compute a negative log likelihood gradient
    
    INPUT VARIABLES:
    - y:           Observed data (Vector: Nx1)
    - tx:          Input data (Matrix: NxD) 
    - w:           Weights (Vector: Dx1)
    - gamma:       Step size for the stoch gradient descent (Scalar/constant)
    
    OUTPUT VARIABLES:
    - loss:        Mean square error of weights w (Scalar)
    - w:           Weights calculated (Vector: Dx1)
    """

    loss = compute_negative_log_likelihood_loss(y,tx,w) + lambda_*np.sum(np.square(w))
    gradient = compute_negative_log_likelihood_gradient(y,tx,w) + 2*lambda_*w
    
    # Updating the w
    w = w - gamma*gradient
    
    return w, loss
