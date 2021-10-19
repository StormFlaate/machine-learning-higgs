import numpy as np
from numpy.core.fromnumeric import shape
from proj1_helpers import batch_iter, compute_gradient, compute_stoch_gradient, compute_loss
from proj1_helpers import logistic_regression_gradient_descent_one_step, compute_gradient_log_likelihood
from proj1_helpers import reguralized_logistic_regression_one_step

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    loss = compute_loss(y,tx,w)
    for n_iter in range(max_iters):
        loss = compute_loss(y,tx,w)
        gradient = compute_gradient(y,tx,w)
        w = w - gamma*gradient

    return w, loss


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    batch_size = 1  # Change this maybe
    w = initial_w
    loss = compute_loss(y,tx,w)
    for y_n, tx_n in batch_iter(y, tx, batch_size, max_iters): # Import function
        loss = compute_loss(y_n,tx_n,w)
        stoch_gradient = compute_stoch_gradient(y_n, tx_n, w)
        w = w - gamma*stoch_gradient
        
    return w, loss

def least_squares(y, tx):
    #w = np.linalg.inv((tx.T)@(tx))@((tx.T)@y)
    w = np.linalg.lstsq(tx,y)[0]
    N = y.shape[0] # Number of elements in vector y
    error = y-(tx@w)
    loss = (1/(2*N))*(error.T @ error)
    
    return w, loss

"""
def ridge_regression(y, tx, lambda_):
    N = y.shape[0] # Number of elements in vector y
    XTX = tx.T@tx 
    lambda_marked = 2*lambda_*N*np.identity(XTX.shape[0]) # calculating the lambda_marked value
    XTy = (tx.T @ y)
    w = (np.linalg.inv(XTX+lambda_marked)) @ XTy #Taking inverse and matmult with second term
    return w
"""

def ridge_regression(y, tx, lambda_): 
    N = tx.shape[0] # Collecting the number of rows in tx
    identity_D = np.identity(tx.shape[1]) # Getting identity matrix from columns of tx
    w = np.linalg.solve(tx.T@tx+2*lambda_*N*identity_D, tx.T@y) # Calculating weights with linalg solve
    loss = compute_loss(y, tx, w) # Computing loss
    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma_):
    w = initial_w # Setting the initial w to w
    loss = float('inf')
    for i in range(max_iters):
        # Loops max_iters number of times, getting last loss and last w each time
        w, loss = logistic_regression_gradient_descent_one_step(y, tx, w, gamma_)
    return w, loss


def reguralized_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma_):
    w = initial_w # Setting the initial w to w
    for i in range(max_iters):
        # Loops max_iters number of times, getting last loss and last w each time
        w, loss = reguralized_logistic_regression_one_step(y, tx, w, gamma_, lambda_)
    return w, loss

