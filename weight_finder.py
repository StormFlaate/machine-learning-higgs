import numpy as np
import matplotlib.pyplot as plt

from implementations import least_squares, least_squares_GD, least_squares_SGD
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

#--------------- Function Uses -----------------#

# Uncomment the following lines to train the different functions
#train_least_squares("data/weights/least_squares_weights.csv")
train_least_squares_SGD("data/weights/least_squares_SGD_weights.csv", np.logspace(-15,10,30))
