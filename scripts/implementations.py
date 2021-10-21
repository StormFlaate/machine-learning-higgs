import numpy as np
from numpy.core.fromnumeric import shape
from proj1_helpers import *

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm using least squares.
    
    INPUT VARIABLES:
    - y:           Observed data (Vector: Nx1)
    - tx:          Input data (Matrix: NxD) 
    - initial_w:   Initial weights (Vector: Dx1)
    - max_iters:   Number of iterations we will run (Scalar/constant)
    - gamma:       Step size for the stoch gradient descent (Scalar/constant)
    
    OUTPUT VARIABLES:
    - loss:        Mean square error of weights w (Scalar)
    - w:           Weights calculated (Vector: Dx1)
    """

    w = initial_w
    for n_iter in range(max_iters):
        # Computing gradient and update w
        gradient = compute_least_squares_gradient(y,tx,w)
        w = w - gamma*gradient
    
    loss = compute_mean_square_error(y,tx,w)

    return w, loss


def least_squares_SGD(y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent algorithm.
    
    INPUT VARIABLES:
    - y:           Observed data (Vector: Nx1)
    - tx:          Input data (Matrix: NxD) 
    - initial_w:   Initial weights (Vector: Dx1)
    - batch_size:  How many (y,tx) pairs will be taken each iteration (Scalar/constant)
    - max_iters:   Number of iterations we will run (Scalar/constant)
    - gamma:       Stepsize for the stoch gradient descent (Scalar/constant)
    
    OUTPUT VARIABLES:
    - loss:        Mean square error of weights w (Scalar)
    - w:           Weights calculated (Vector: Dx1)
    """
    
    w = initial_w
    for n_iter in range(max_iters):
        # Taking a batch_size sized batch out of the y and tx
        # These are taken out at random out of the total y and tx
        for y_n, tx_n in batch_iter(y,tx, batch_size):
            # Computes the batch gradient: 
            # In Stochastic gradient we have batch_size = 1
            grad = compute_least_squares_stoch_gradient(y_n, tx_n, w)
            w = w - gamma*grad
        
    loss = compute_mean_square_error(y,tx,w)
        
    return w, loss


def least_squares(y, tx):
    """Least squares algorithm.
    
    INPUT VARIABLES:
    - y:           Observed data (Vector: Nx1)
    - tx:          Input data (Matrix: NxD) 
    
    OUTPUT VARIABLES:
    - loss:        Mean square error of weights w (Scalar)
    - w:           Weights calculated (Vector: Dx1)
    """
    w = np.linalg.lstsq(tx,y,rcond=None)[0]
    loss = compute_mean_square_error(y, tx,w)
    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent
    
    INPUT VARIABLES:
    - y:           Observed data (Vector: Nx1)
    - tx:          Input data (Matrix: NxD) 
    - w:           Weights (Vector: Dx1)
    - max_iters:   Number of steps/iterations we will do with the gradient descent
    - gamma:       Step size for the stoch gradient descent (Scalar/constant)
    
    OUTPUT VARIABLES:
    - loss:        Mean square error of weights w (Scalar)
    - w:           Weights calculated (Vector: Dx1)
    """
    w = initial_w # Setting the weight to the initial weight

    for i in range(max_iters):
        # Loops max_iters number of times, getting last loss and last w each time
        w, loss = logistic_regression_gradient_descent_one_step(y, tx, w, gamma)
    return w, loss

def logistic_regression_SGD(y, tx, initial_w, batch_size, max_iters, gamma):
    """Logistic regression using gradient descent
    
    INPUT VARIABLES:
    - y:           Observed data (Vector: Nx1)
    - tx:          Input data (Matrix: NxD) 
    - w:           Weights (Vector: Dx1)
    - batch_size:  Number of elements that will be used per iteration for the stoch gradient descent
    - max_iters:   Number of steps/iterations we will do with the stoch gradient descent
    - gamma:       Step size for the stoch gradient descent (Scalar/constant)
    
    OUTPUT VARIABLES:
    - loss:        Mean square error of weights w (Scalar)
    - w:           Weights calculated (Vector: Dx1)
    """
    w = initial_w # Setting the weight to the initial weight

    for i in range(max_iters):
        # Loops max_iters number of times, getting last loss and last w each time
        w, loss = logistic_regression_stoch_gradient_descent_one_step(y, tx, w, batch_size, gamma)
    return w, loss

def reguralized_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    w = initial_w # Setting the weight to the initial weight

    for i in range(max_iters):
        # Loops max_iters number of times, getting last loss and last w each time
        w, loss = penalized_logistic_regression_gradient_descent_one_step(y, tx, w, gamma, lambda_)
    return w, loss

