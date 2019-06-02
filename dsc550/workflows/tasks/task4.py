#%%
import pickle
import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

data_path = Path('data')
model_path = Path('models')
report_path = Path('reports')
source_path = data_path / 'source/reddit/'
interim_path = data_path / 'interim/'
processed_path = data_path / 'processed/'

def task4(param1):
    """Task 4 takes the testing data form Task 2 and the model data from Task 3
    and creates a report of a few metrics for analysis of the model.

    Args:
        param1 (str): Filename to process

    Returns:
        bool: True if the process succeeds in writing the output files,
        False if it fails.
    """

    # Load Model
    pickle_in_model_path = model_path / (param1 + '_model.pkl')
    pickle_in_model = open(pickle_in_model_path, 'rb')
    model = pickle.load(pickle_in_model)
    pickle_in_model.close()

    # Load X_Testing Data
    pickle_in_X_test_path = interim_path / (param1 + '_X_test.pkl')
    pickle_in_features = open(pickle_in_X_test_path, 'rb')
    X_test = pickle.load(pickle_in_features)
    pickle_in_features.close()

    # Load y_Testing Data
    pickle_in_y_test_path = interim_path / (param1 + '_y_test.pkl')
    pickle_in_targets = open(pickle_in_y_test_path, 'rb')
    y_test = pickle.load(pickle_in_targets)
    pickle_in_targets.close()

    # Use best model and test data for final evaluation
    y_pred = model.predict(X_test)

    # Calculate the ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)

    # Assign report path and name
    md_path_report = report_path / (param1 + '_report.md')

    try:
        report = open(md_path_report, "w")

        print('#### Results for', param1, file=report)
        print('', file=report)
        print('Best Model:', model.best_estimator_.get_params()['classifier'], file=report)
        print('', file=report)
        print('Best Score:', model.best_score_, file=report)
        print('Best Penalty:', model.best_estimator_.get_params()['classifier__penalty'], file=report)
        print('Best C:', model.best_estimator_.get_params()['classifier__C'], file=report)
        print('Best AUC:', auc(fpr, tpr), file=report)
        print('', file=report)
        print(classification_report(y_test, y_pred), file=report)
        print('', file=report)
        print('Confusion Matrix:', file=report)
        print(confusion_matrix(y_test, y_pred), file=report)

        report.close()

    except:
        return False

    else:
        pickle_in_X_test_path.replace(processed_path / (param1 + '_X_test.pkl'))
        pickle_in_y_test_path.replace(processed_path / (param1 + '_y_test.pkl'))
        return True

#%%
task4('controversial-comments')

#%%
