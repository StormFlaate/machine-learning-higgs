import numpy as np
from proj1_helpers import load_csv_data

def row_cleaner(data, verbose = False):
    '''Function to remove all rows containing any values equal to -999
       
       
    INPUT VARIABLES:
    - data                     Data set with the jet number column and other "bad" columns removed
    - verbose                  If verbose == True then the program will print updates
    
    OUTPUT VARIABLES:
    - data                     The array tX with all rows removed which contained at least one entry equal to -999
    '''
        
    # Storing the initial number of columns for future comparison in case that verbose == True
    if verbose:
        len_initial = len(data)
    
    # Removing all rows where there are any entries equal to -999
    data = data[np.where(np.all(data != -999, 1))]
    
    # If verbose == True, printing the number of removed rows (and the percentage of data that this corresponds to)
    if verbose:
        print("Removed a total of ", len_initial - len(data), " rows, equating to removing ", 100 * (1 - len(data) / len_initial), " % of the data")
    
    return data


def data_maker(DATA_TRAIN_PATH, DATA_TEST_PATH, DATA_SAVE_PATH, verbose = False):
    ''' Function to read, concatenate, and clean the data before writing it to a file.
    
    INPUT VARIABLES:
    - DATA_TRAIN_PATH          Path to the file containing the training data
    - DATA_TEST_PATH           Path to the file containing the test data
    - DATA_SAVE_PATH           Location to which the processed data is to be saved
    - verbose                  If verbose == True then the program will print updates during data cleaning
    
    OUTPUT VARIABLES:
    - None                     The cleaned data is written to a file but there is no other output
    
    '''
    
    y_train, tX_train, ids_train = load_csv_data(DATA_TRAIN_PATH)
    y_test, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)
    
    # Concatenate the data sets. As cross validation will be use it is not necessary to have distinct train and test sets
    y = np.vstack((np.reshape(y_train, (len(y_train), 1)), np.reshape(y_test, (len(y_test), 1))))
    tX = np.vstack((tX_train, tX_test))
    ids = np.vstack((np.reshape(ids_train, (len(ids_train), 1)), np.reshape(ids_test, (len(ids_test), 1))))
    
    # Removing the column with the jet numbers and the columns where all entries for jet numbers 0 and 1 equal -999
    #      -> These columns where found by consulting the Cern documentation for the data
    #      -> The column with the jet numbers has index 22
    #      -> The columns with many -999 entries are those with indices 4,  5,  6, 12, 23, 24, 25, 26, 27, and 28
    
    tX = np.hstack((tX[:,:4], tX[:,7:12], tX[:,13:22], tX[:,29:]))
    
    # Concatenating tX, y, and ids (in that order) such that the same rows will be removed in all of them
    
    data = np.hstack((ids, y, tX))
    
    # Removing all rows with corrupted (-999) entries in them
    
    data = row_cleaner(data, verbose)
    
    # Writing the data to the file. The order of the data in the original file is preserved (the entries are in the order ids - y - tX)
    # NOTE: the y are now directly in the numerical format [-1, 1]
    
    with open(DATA_SAVE_PATH, 'w') as f:
        for row in data:
            for item in row:
                f.write("%s," % item)
            f.write("\n")
    if verbose:
        print("file_written")

    return None

data_maker("../data/train.csv", "../data/test.csv", "../data/clean_data.csv", True)