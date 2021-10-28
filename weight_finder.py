import numpy as np
import matplotlib.pyplot as plt

from implementations import least_squares, least_squares_GD, least_squares_SGD, ridge_regression, logistic_regression_SGD, reguralized_logistic_regression
from weight_helpers import *
from data_reader import load_clean_csv


#--------------- Function Implementations -----------------#
def train_least_squares(DATA_SAVE_PATH, TRAIN_DATA_PATH = "data/clean_train_data.csv"):
    y, tX, ids = load_clean_csv(TRAIN_DATA_PATH)

    w, _ = least_squares(y, tX)

    weight_writer(w, DATA_SAVE_PATH)


def train_least_squares_GD(DATA_SAVE_PATH, gamma_array, TRAIN_DATA_PATH = "data/clean_train_data.csv"):
    print("now training least squares GD")

    y, tX, ids = load_clean_csv(TRAIN_DATA_PATH)

    test_loss = []
    for g in gamma_array:
        _, test_ls = cross_validation(y, tX, 3, least_squares_GD, g)
        if np.isnan(test_ls): test_ls = np.array([[100]])
        if np.isinf(test_ls): test_ls = np.array([[100]])
        if test_ls > 101: test_ls = np.array([[100]])
        test_loss.append(test_ls[0,0])
        print("finished round ", np.where(gamma_array==g))

    g_star = gamma_array[np.argmin(test_loss)]
    print("hi", test_loss)

    print(g_star)

    w, _ = least_squares_GD(y, tX, g_star)

    weight_writer(w, DATA_SAVE_PATH)


def train_least_squares_SGD(DATA_SAVE_PATH, gamma_array, TRAIN_DATA_PATH = "data/clean_train_data.csv"):
    print("now training least squares SGD")
    y, tX, ids = load_clean_csv(TRAIN_DATA_PATH)

    test_loss = []
    for g in gamma_array:
        _, test_ls = cross_validation(y, tX, 3, least_squares_SGD, g)
        if np.isnan(test_ls): test_ls = np.array([[10000]])
        if np.isinf(test_ls): test_ls = np.array([[10000]])
        test_loss.append(test_ls[0,0])
        print("finished round ", np.where(gamma_array==g)[0])

    g_star = gamma_array[np.argmin(test_loss)]
    print("hi", test_loss)

    print(g_star)
    plt.semilogx(gamma_array, test_loss)
    plt.show()

    w, _ = least_squares_SGD(y, tX, g_star)
    print(w)
    #weight_writer(w, DATA_SAVE_PATH)

def train_ridge_regression(DATA_SAVE_PATH, lambda_array, TRAIN_DATA_PATH = "data/clean_train_data.csv"):
    print("now training ridge regression")
    y, tX, ids = load_clean_csv(TRAIN_DATA_PATH)

    test_loss = []
    for l in lambda_array:
        _, test_ls = cross_validation(y, tX, 3, ridge_regression, l)
        if np.isnan(test_ls): test_ls = np.array([[10000]])
        if np.isinf(test_ls): test_ls = np.array([[10000]])
        test_loss.append(test_ls[0,0])
        print("finished round ", np.where(lambda_array==l)[0])

    l_star = lambda_array[np.argmin(test_loss)]
    print("hi", test_loss)

    print(l_star)
    plt.semilogx(lambda_array, test_loss)
    plt.show()

    w, _ = ridge_regression(y, tX, l_star)
    print(w)
    weight_writer(w, DATA_SAVE_PATH)


def train_logistic_SGD(DATA_SAVE_PATH, gamma_array, TRAIN_DATA_PATH = "data/clean_train_data.csv"):
    print("now training logistic SGD")
    y, tX, ids = load_clean_csv(TRAIN_DATA_PATH)

    test_loss = []
    for g in gamma_array:
        _, test_ls = cross_validation(y, tX, 3, logistic_regression_SGD, g)
        if np.isnan(test_ls): test_ls = np.inf
        test_loss.append(test_ls[0,0])
        print("finished round ", np.where(gamma_array==g)[0])

    g_star = gamma_array[np.argmin(test_loss)]
    print("hi", test_loss)

    print(g_star)
    plt.semilogx(gamma_array, test_loss)
    plt.show()

    w, _ = logistic_regression_SGD(y, tX, g_star)
    print(w)
    weight_writer(w, DATA_SAVE_PATH)


def train_regularized_logistic_SGD(DATA_SAVE_PATH, lambda_array, gamma_array, TRAIN_DATA_PATH = "data/clean_short_train.csv"):
    print("now training regularized logistic SGD")
    y, tX, ids = load_clean_csv(TRAIN_DATA_PATH)

    test_loss = []
    for l in lambda_array:
        loss_saver = []
        for g in gamma_array:
            _, test_ls = cross_validation(y, tX, 3, reguralized_logistic_regression, l, g)
            if np.isnan(test_ls): test_ls = np.array([[np.inf]])
            loss_saver.append(test_ls[0,0])
            print("finished sub_round ", np.where(gamma_array==g)[0], "-", np.where(lambda_array==l)[0])

        test_loss.append(loss_saver)
        print("finished round ", np.where(lambda_array==l)[0])

    location = np.argwhere(test_loss == min(np.array(test_loss).flatten()))[0]
    print(location)
    l_star = lambda_array[location[0]]
    g_star = gamma_array[location[1]]
    print("hi", test_loss)

    print(l_star)
    print(g_star)
    plt.semilogx(gamma_array, test_loss[0][:])
    plt.show()
    plt.semilogx(lambda_array, test_loss[:][0])
    plt.show()

    w, _ = reguralized_logistic_regression(y, tX, l_star, g_star)
    print(w)
    weight_writer(w, DATA_SAVE_PATH)

#--------------- Function Uses -----------------#

# Uncomment the following lines to train the different functions
#train_least_squares("data/weights/least_squares_weights.csv")
#train_least_squares_GD("data/weights/least_squares_GD_weights.csv", np.logspace(-5.1,-4.7,5))
#train_least_squares_SGD("data/weights/least_squares_SGD_weights.csv", np.logspace(-10.2,-9,5))
#train_ridge_regression("data/weights/ridge_regression_weights.csv", np.logspace(-9,-3,10))
#train_logistic_SGD("data/weights/logistic_SGD_weights.csv", np.logspace(-6, -4.5, 5))
train_regularized_logistic_SGD("data/weights/regularized_logistic_SGD_weights_short.csv", np.logspace(4.5, 6.5, 5), np.logspace(-8, -6.5, 5))
