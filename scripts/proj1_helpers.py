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



# Hellper functions for implementation.py ==========================================================
def compute_stoch_gradient(y, tx, w):
    N = y.shape[0] # Number of elements in vector y
    error = y-(tx@w)
    return -(1/N)*(tx.T)@error

def compute_gradient(y, tx, w):
    N = y.shape[0] # Number of elements in vector y
    error = y-(tx@w)
    return -(1/N)*(tx.T)@error

def compute_loss(y, tx, w):
    N = y.shape[0] # Number of elements in vector y
    error = y-(tx@w)
    return (1/(2*N))*(error.T@error)

def logistic_regression_gradient_descent_one_step(y, tx, w, gamma_):
    loss = compute_loss_log_likelihood(y, tx, w)
    gradient = compute_gradient_log_likelihood(y, tx, w)
    w = w - gamma_ * gradient
    return loss, w

def sigmoid_function(a):
    sig = 1/(1+np.exp(-a)) #Using np.exp instead of math.exp for speed
    return sig

def compute_loss_log_likelihood(y, tx, w):
    y_ = sigmoid_function(tx@w)
    first_term = -y.T@np.log(y_)
    second_term = -(1-y).T@np.log(1-y_)
    loss =  first_term + second_term
    return np.squeeze(loss)

def compute_gradient_log_likelihood(y, tx, w):
    error = sigmoid_function(tx@w) - y
    return  tx.T@error

def reguralized_logistic_regression_one_step(y, tx, w, gamma_, lambda_):
    init_loss = compute_loss_log_likelihood(y, tx, w) # Intial loss
    loss = init_loss+lambda_*(w.T@w) # Adding the regularization
    gradient = compute_gradient_log_likelihood(y, tx, w) + 2*w*lambda_
    w = w - gamma_*gradient
    return loss, w
# ============================================================================
