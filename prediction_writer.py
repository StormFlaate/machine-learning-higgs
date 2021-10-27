import numpy as np
from data_reader import load_weights, load_clean_csv

# Reading the weights from file
def prediction_writer(TEST_DATA_PATH, WEIGHTS_PATH, DATA_SAVE_PATH):

    # Loading the weights
    w = load_weights(WEIGHTS_PATH)

    # Loading the test data
    _, tX, ids = load_clean_csv(TEST_DATA_PATH, False)

    # generating predictions
    y_good_guess = tX @ w

    # discretizing into -1 and 1
    y_good_guess[np.where(y_good_guess < .5)] = -1
    y_good_guess[np.where(y_good_guess >= .5)] = 1

    # Write the prediction file in the desired format
    with open(DATA_SAVE_PATH, 'w') as f:
        f.write("Id,Prediction\n")
        for i in range(len(y_good_guess)):
            f.write(str(ids[i]) + "," + str(y_good_guess[i,0]) + "\n")
            if i%10000 == 1: print(i, y_good_guess[i,0], y_good_guess[i])

    print(len(y_good_guess))

prediction_writer("data/clean_test_data.csv", "data/weights/least_squares_weights.csv", "data/predictions/least_squares_predictions.csv")