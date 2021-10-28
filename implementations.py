import numpy as np
from proj1_helpers import *

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


def least_squares_GD(y, tx, gamma, initial_w=False, max_iters=500):
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

    if not initial_w:
        initial_w = np.ones((len(tx[0, :]),1))/2

    w = initial_w
    for n_iter in range(max_iters):
        # Computing gradient and update w

        gradient = compute_least_squares_gradient(y,tx,w)
        w = w - gamma*gradient
        if n_iter%10 == 0: print(n_iter/max_iters)

    loss = compute_mean_square_error(y,tx,w)

    return w, loss


def least_squares_SGD(y, tx, gamma, initial_w=False, max_iters=50000, batch_size=1):
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

    if not initial_w:
        initial_w = np.ones((len(tx[0, :]),1))/2

    w = initial_w
    epoch = len(y)
    for n_iter in range(max_iters):
        # Taking a batch_size sized batch out of the y and tx
        # These are taken out at random out of the total y and tx

        n_iter = n_iter%epoch
        x_n = tx[n_iter, :]
        y_n = y[n_iter]
        grad = np.reshape(x_n, (len(x_n), 1)) * (y_n - x_n @ w)
        w = w - gamma*grad
        if n_iter%100 == 0: print(n_iter/max_iters)
    loss = compute_mean_square_error(y,tx,w)

    return w, loss


def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    n = len(y)
    y = np.reshape(y, (len(y), 1))
    gm = tx.T@tx
    lambda_i = 2*lambda_*len(y)*np.identity(len(gm))
    w = (np.linalg.inv(gm+lambda_i))@(tx.T@y)
    loss = compute_mean_square_error(y,tx,w)
    return w, loss



def logistic_regression_SGD(y, tx, gamma, initial_w=None, max_iters=50000):
    """Logistic regression using gradient descent

    INPUT VARIABLES:
    - y:           Observed data (Vector: Nx1)
    - tx:          Input data (Matrix: NxD)
    - initial_w:   Initial weights (Vector: Dx1)
    - batch_size:  Number of elements that will be used per iteration for the stoch gradient descent
    - max_iters:   Number of steps/iterations we will do with the stoch gradient descent
    - gamma:       Step size for the stoch gradient descent (Scalar/constant)

    OUTPUT VARIABLES:
    - loss:        Mean square error of weights w (Scalar)
    - w:           Weights calculated (Vector: Dx1)
    """

    if not initial_w:
        initial_w = np.ones((len(tx[0, :]),1))/2

    w = initial_w # Setting the weight to the initial weight
    y = np.reshape(y, (len(y), 1))
    epoch = len(y)

    for i in range(max_iters):
        if i%1000 == 0: print(i/max_iters)
        i = i%epoch
        # Loops max_iters number of times, getting last loss and last w each time
        y_n = y[i]
        tx_n = tx[i]
        gradient = np.reshape(tx_n, (len(tx_n),1)) * (sigmoid(tx_n @ w)-y_n)
        w = w - gamma*gradient

    loss = compute_negative_log_likelihood_loss(y, tx, w)

    return w, loss


def reguralized_logistic_regression(y, tx, lambda_, gamma, initial_w=None, max_iters=50000):
    """Reguralized logistic regression using gradient descent

    INPUT VARIABLES:
    - y:           Observed data (Vector: Nx1)
    - tx:          Input data (Matrix: NxD)
    - lambda_:     Regularization parameter
    - initial_w:   Initial weights (Vector: Dx1)
    - max_iters:   Number of steps/iterations we will do with the stoch gradient descent
    - gamma:       Step size for the stoch gradient descent (Scalar/constant)


    OUTPUT VARIABLES:
    - loss:        Mean square error of weights w (Scalar)
    - w:           Weights calculated (Vector: Dx1)
    """

    if not initial_w:
        initial_w = np.ones((len(tx[0, :]),1))/2

    w = initial_w # Setting the weight to the initial weight
    y = np.reshape(y, (len(y), 1))
    epoch = len(y)

    for i in range(max_iters):
        if i%1000 == 0: print(i/max_iters)
        i = i%epoch
        # Loops max_iters number of times, getting last loss and last w each time
        y_n = y[i]
        tx_n = tx[i]
        gradient = np.reshape(tx_n, (len(tx_n),1)) * (sigmoid(tx_n @ w)-y_n) + lambda_*w #TODO: discuss this. Is the derivative this simple? Also removed 2 as it cancels with the 1/2
        w = w - gamma*gradient

    loss = compute_negative_log_likelihood_loss(y, tx, w)

    return w, loss
