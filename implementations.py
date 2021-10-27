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


def least_squares_GD(y, tx, gamma, initial_w=False, max_iters=5000):
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
        initial_w = np.ones(len(tx[0, :]))/2

    w = initial_w
    for n_iter in range(max_iters):
        # Computing gradient and update w
        gradient = compute_least_squares_gradient(y,tx,w)
        w = w - gamma*gradient

    loss = compute_mean_square_error(y,tx,w)

    return w, loss


def least_squares_SGD(y, tx, gamma, initial_w=False, max_iters=500000, batch_size=1):
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
        initial_w = np.ones(len(tx[0, :]))/2

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
