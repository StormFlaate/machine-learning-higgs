import numpy as np
import matplotlib.pyplot as plt

from implementations import least_squares, least_squares_GD, least_squares_SGD, logistic_regression, reguralized_logistic_regression, ridge_regression
from proj1_helpers import normalize
from weight_helpers import *
from data_reader import load_clean_csv


#--------------- Function Implementations -----------------#
def train_least_squares(DATA_SAVE_PATH, TRAIN_DATA_PATH = "data/clean_train_data.csv"):
    y, tX, ids = load_clean_csv(TRAIN_DATA_PATH)

    w, _ = least_squares(y, tX)

    weight_writer(w, DATA_SAVE_PATH)


def train_least_squares_SGD(DATA_SAVE_PATH, gamma_array, TRAIN_DATA_PATH = "data/clean_train_data.csv"):
    y, tX, ids = load_clean_csv(TRAIN_DATA_PATH)

    test_loss = []
    for g in gamma_array:
        test_ls = cross_validation(y, tX, 10, least_squares_SGD, g)
        test_loss.append(test_ls)

    g_star = np.argmin(test_loss)

    print(g_star)
    plt.semilogx(gamma_array, test_loss)
    plt.show()

    w, _ = least_squares_SGD(y, tX, g_star)

    weight_writer(w, DATA_SAVE_PATH)

    pass


def train_logistic_regression(DATA_SAVE_PATH, gamma_array, TRAIN_DATA_PATH = "../data/clean_train_data.csv"):
    y, tX, ids = load_clean_csv(TRAIN_DATA_PATH)
    
    y, tX = normalize(y), normalize(tX)
    np.random.seed(1)
    # Split data into training and testing
    zipped_shuffle = list(zip(y, tX))
    np.random.shuffle(zipped_shuffle)
    y, tX = zip(*zipped_shuffle)
    y = np.array(y)
    tX = np.array(tX)
    

    y_train, tX_train = y[:int(len(y)*0.9)], tX[:int(len(tX)*0.9)]
    y_test, tX_test = y[int(len(y)*0.9):], tX[int(len(tX)*0.9):]

    max_iters = 1000


    best_loss, best_gamma = (float('inf'), 1) # best loss, and gamma value init values
    for gamma in gamma_array:
        print("Current gamma: ", gamma)
        w = np.zeros(19)
        w, test_ls = logistic_regression(y_train, tX_train, w, max_iters, gamma)


        y_good_guess = normalize(tX_test) @ w
        # discretizing into -1 and 1
        y_good_guess[np.where(y_good_guess < .5)] = -1
        y_good_guess[np.where(y_good_guess >= .5)] = 1

        y_test[np.where(y_test < .5)] = -1
        y_test[np.where(y_test >= .5)] = 1
        accuracy = (y_test == y_good_guess).mean()
        print(accuracy)

        if test_ls < best_loss:
            best_loss = test_ls
            best_gamma = gamma


    g_star = best_gamma

    print("Best gamma value: ", g_star)
    
    w = np.zeros(19)
    max_iters = 1000
    print("Starting training!")
    w, _ = logistic_regression(y, tX, w, max_iters, g_star)

    weight_writer(w, DATA_SAVE_PATH)


def train_reguralized_logistic_regression(DATA_SAVE_PATH, lambda_array, gamma_array, TRAIN_DATA_PATH = "../data/clean_train_data.csv"):
    y, tX, ids = load_clean_csv(TRAIN_DATA_PATH)
    y, tX = normalize(y), normalize(tX)
    np.random.seed(1)
    # Split data into training and testing
    zipped_shuffle = list(zip(y, tX))
    np.random.shuffle(zipped_shuffle)
    y, tX = zip(*zipped_shuffle)
    y = np.array(y)
    tX = np.array(tX)
    

    y_train, tX_train = y[:int(len(y)*0.9)], tX[:int(len(tX)*0.9)]
    y_test, tX_test = y[int(len(y)*0.9):], tX[int(len(tX)*0.9):]

    max_iters = 500
    gamma_array = gamma_array[7:9]
    best_accuracy, best_gamma, best_lambda = 0.0, float('inf'), 1 # best loss, and gamma value init values
    for gamma in gamma_array:
        for lambda_ in lambda_array:
            print(f"Current gamma: {gamma} | lambda: {lambda_}")


            w = np.zeros(19)
            w, test_ls = reguralized_logistic_regression(y_train, tX_train, lambda_, w, max_iters, gamma)


            y_good_guess = normalize(tX_test) @ w
            # discretizing into -1 and 1
            y_good_guess[np.where(y_good_guess < .5)] = -1
            y_good_guess[np.where(y_good_guess >= .5)] = 1

            y_test[np.where(y_test < .5)] = -1
            y_test[np.where(y_test >= .5)] = 1
            accuracy = (y_test == y_good_guess).mean()
            
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_gamma = gamma
                best_lambda = lambda_


    g_star = best_gamma
    l_star = best_lambda

    print("Best gamma value: ", g_star)
    print("Best lambda value: ", l_star)
    
    w = np.zeros(19)
    max_iters = 1000
    print("Starting training!")
    w, _ = reguralized_logistic_regression(y, tX, l_star, w, max_iters, g_star)

    weight_writer(w, DATA_SAVE_PATH)


#--------------- Function Uses -----------------#

# Uncomment the following lines to train the different functions
#train_least_squares("data/weights/least_squares_weights.csv")
#train_least_squares_SGD("data/weights/least_squares_SGD_weights.csv", np.logspace(-15,10,30))
train_logistic_regression("../data/weights/logistic_regression_weights.csv", np.logspace(-15,-1,15))
#train_reguralized_logistic_regression("../data/weights/regurlized_logistic_regression_weights_given_gamma.csv", np.logspace(-15,10,4), np.logspace(-15,-1,15))

