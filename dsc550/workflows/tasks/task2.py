# Model Data Splitting

#%%
import pickle
import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.model_selection import train_test_split

data_path = Path('data')
model_path = Path('models')
source_path = data_path / 'source/reddit/'
interim_path = data_path / 'interim/'
processed_path = data_path / 'processed/'

def task2(param1):
    """Task 2 splits the Features and Targets from Task 1 into 4 datasets.
    X_train
    X_test
    y_train
    y_test

    Args:
        param1 (str): Filename to process

    Returns:
        bool: True if the process succeeds in writing the output files,
        False if it fails.
    """

    # Load Features
    pickle_in_features_path = interim_path / (param1 + '_features.pkl')
    pickle_in_features = open(pickle_in_features_path, 'rb')
    features = pickle.load(pickle_in_features)
    pickle_in_features.close()

    # Load Targets
    pickle_in_targets_path = interim_path / (param1 + '_targets.pkl')
    pickle_in_targets = open(pickle_in_targets_path, 'rb')
    targets = pickle.load(pickle_in_targets)
    pickle_in_targets.close()

    # Split Data for Testing and Training
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.25)

    pickle_path_model_X_train = interim_path / (param1 + '_X_train.pkl')
    pickle_path_model_y_train = interim_path / (param1 + '_y_train.pkl')
    pickle_path_model_X_test = interim_path / (param1 + '_X_test.pkl')
    pickle_path_model_y_test = interim_path / (param1 + '_y_test.pkl')

    # Pickle Data for Testing and Training
    try:
        try:
            pickle_out_model_X_train = open(pickle_path_model_X_train, "wb")
            pickle.dump(X_train, pickle_out_model_X_train)
            pickle_out_model_X_train.close()
        except:
            print("X_train Pickle Failed")

        try:
            pickle_out_model_y_train = open(pickle_path_model_y_train, "wb")
            pickle.dump(y_train, pickle_out_model_y_train)
            pickle_out_model_y_train.close()
        except:
            print("y_train Pickle Failed")

        try:
            pickle_out_model_X_test = open(pickle_path_model_X_test, "wb")
            pickle.dump(X_test, pickle_out_model_X_test)
            pickle_out_model_X_test.close()
        except:
            print("X_test Pickle Failed")

        try:
            pickle_out_model_y_test = open(pickle_path_model_y_test, "wb")
            pickle.dump(y_test, pickle_out_model_y_test)
            pickle_out_model_y_test.close()
        except:
            print("y_test Pickle Failed")

    except:
        return False

    else:
        pickle_in_features_path.replace(processed_path / (param1 + '_features.pkl'))
        pickle_in_targets_path.replace(processed_path / (param1 + '_targets.pkl'))
        return True

#%%
task2('controversial-comments')

#%%
