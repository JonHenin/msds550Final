# Text Classification

#%%
import pickle
import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

data_path = Path('data')
model_path = Path('models')
source_path = data_path / 'source/reddit/'
interim_path = data_path / 'interim/'
processed_path = data_path / 'processed/'

def task3(param1):
    """Task 3 takes the training data from Task 2 and runs through it through
    a data model pipeline. Here we are testing Naive Bayes (MultinomialNB) and
    Logistic Regression with both and L1 and L2 penalty and some hyperparameter
    optimization searching.  The resulting model is outputted to the model folder.

    Args:
        param1 (str): Filename to process

    Returns:
        bool: True if the process succeeds in writing the output files,
        False if it fails.
    """

    # Load X_Training Data
    pickle_in_X_train_path = interim_path / (param1 + '_X_train.pkl')
    pickle_in_features = open(pickle_in_X_train_path, 'rb')
    features = pickle.load(pickle_in_features)
    pickle_in_features.close()

    # Load y_Training Data
    pickle_in_y_train_path = interim_path / (param1 + '_y_train.pkl')
    pickle_in_targets = open(pickle_in_y_train_path, 'rb')
    targets = pickle.load(pickle_in_targets)
    pickle_in_targets.close()

    # Create a pipeline
    pipe = Pipeline([("classifier", LogisticRegression())])

    # Create dictionary with candidate learning algorithms and their hyperparameters
    search_space = [{"classifier": [LogisticRegression()],
                     "classifier__penalty": ['l1', 'l2'],
                     "classifier__C": np.logspace(0, 4, 10)},
                    {"classifier": [MultinomialNB()]}]

    # Create grid search
    gridsearch = GridSearchCV(pipe, search_space, cv=3, verbose=0)

    # Fit grid search
    model = gridsearch.fit(features, targets)

    # Assign pickle path and param1 for model and training data.
    pickle_path_model = model_path / (param1 + '_model.pkl')

    try:
        # Create / Open pickle files
        pickle_out_model = open(pickle_path_model, "wb")
        pickle.dump(model, pickle_out_model)
        pickle_out_model.close()

    except:
        return False

    else:
        pickle_in_X_train_path.replace(processed_path / (param1 + '_X_train.pkl'))
        pickle_in_y_train_path.replace(processed_path / (param1 + '_y_train.pkl'))
        return True

#%%
task3('controversial-comments')

#%%
