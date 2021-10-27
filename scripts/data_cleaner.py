import numpy as np
from proj1_helpers import load_csv_data

def row_cleaner(data):
    '''Function to remove all rows containing any values equal to -999
          
    INPUT VARIABLES:
    - data                     Data set with the jet number column and other "bad" columns removed
    OUTPUT VARIABLES:
    - data                     The array tX with all rows removed which contained at least one entry equal to -999
    '''

    #Finding the modal values
    modes = np.mean(data, 1)

    # Removing all rows where there are any entries equal to -999
    data = data[np.where(np.all(data != -999, 1))]

    return data


def data_maker(DATA_PATH, DATA_SAVE_PATH, verbose=False, length=None):
    ''' Function to read, concatenate, and clean the data before writing it to a file.
    
    INPUT VARIABLES:
    - DATA_PATH          Path to the file containing the relevant data
    - DATA_SAVE_PATH           Location to which the processed data is to be saved
    - verbose                  If verbose == True then the program will print updates during data cleaning
    
    OUTPUT VARIABLES:
    - None                     The cleaned data is written to a file but there is no other output
    
    '''
    
    y, tX, ids = load_csv_data(DATA_PATH)

    # Abbreviating length to make a short dataset if desired
    if length:
        y = y[:length]
        tX = tX[:length]
        ids = ids[:length]
    
    # Reshaping y and ids to be column vectors
    y = np.reshape(y, (len(y), 1))
    ids = np.reshape(ids, (len(ids), 1))
    
    # Removing the column with the jet numbers and the columns where all entries for jet numbers 0 and 1 equal -999
    #      -> These columns where found by consulting the Cern documentation for the data
    #      -> The column with the jet numbers has index 22
    #      -> The columns with many -999 entries are those with indices 4,  5,  6, 12, 23, 24, 25, 26, 27, and 28
    
    tX = np.hstack((tX[:,:4], tX[:,7:12], tX[:,13:22], tX[:,29:]))

    # Concatenating tX, y, and ids (in that order) such that the same rows will be removed in all of them

    data = np.hstack((ids, y, tX))

    # Removing all rows with corrupted (-999) entries in them
    
    data_no999 = row_cleaner(data)

    # Regressing the column corresponding to DER_mass_vis and the one corresponding to DER_mass_MMC
    a = np.hstack((np.ones((len(data_no999),1)), np.reshape(data_no999[:,4], (len(data_no999),1))))
    b = data_no999[:,0]
    weights = np.linalg.lstsq(a, b)[0]

    # Finding -999 values in the first column (only column left with faulty values) and replacing them with regressed values
    locations = np.argwhere(tX[:,0]==-999)
    new_values = np.hstack((np.ones((len(locations),1)), np.reshape(data[locations,4], (len(locations),1)))) @ weights

    data[locations, 2] = np.reshape(new_values, (len(new_values),1))
    
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

data_maker("../data/train.csv", "../data/clean_train_data.csv", True)
data_maker("../data/train.csv", "../data/clean_short_train.csv", True, 10000)

data_maker("../data/test.csv", "../data/clean_test_data.csv", True)
data_maker("../data/test.csv", "../data/clean_short_test.csv", True, 10000)