# Preprocessing

#%%
import pandas as pd
import re
import string
import pickle

from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer

data_path = Path('data')
source_path = data_path / 'source/reddit/'
interim_path = data_path / 'interim/'
processed_path = data_path / 'processed/'


def my_preprocessor(text):
    text = re.sub("\S*\d\S*", "", text).strip() # Strip out any numbers
    text = text.translate(str.maketrans('','', string.punctuation)) # Strip out punctuation
    return text.lower() # Return lowercase values


def task1(param1):
    """Task 1 cleans up the data file in the source/reddit folder,
    splits the data into vectorized features and targets and then outputs a pickle
    file of each.

    Args:
        param1 (str): Filename to process

    Returns:
        bool: True if the process succeeds in writing the output files,
        False if it fails.
    """

    file_path = source_path / (param1 + '.jsonl')

    # Import file into a dataframe
    df = pd.read_json(file_path, lines=True, orient='columns')

    # Create vectorized features
    vectorizer = TfidfVectorizer(preprocessor=my_preprocessor,
                    stop_words='english')
    features = vectorizer.fit_transform(df['txt'])

    # Create targets
    targets = df['con']

    # Assign pickle path and param1
    pickle_path_features = interim_path / (param1 + '_features.pkl')
    pickle_path_targets = interim_path / (param1 + '_targets.pkl')

    # Create / Open pickle files
    pickle_out_features = open(pickle_path_features, "wb")
    pickle_out_targets = open(pickle_path_targets, "wb")

    try:
        pickle.dump(features, pickle_out_features)
        pickle.dump(targets, pickle_out_targets)
        pickle_out_features.close()
        pickle_out_targets.close()
    except:
        return False
    else:
        return True

#%%
task1('controversial-comments')

