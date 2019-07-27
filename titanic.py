"""
titanic.py

Not sure what I'll do here exactly yet, I need to run an exploratory analysis, feature engineer, and
build models.  The last two I might do often more than once, so I'll need to figure out a good way
to structure this project.

Author: Mark Xavier
"""

import pandas as pd

TRAIN_FILE = r'inputs/train.csv'
TEST_FILE = r'inputs/test.csv'

def main():
    """
    """

    training_set = pd.read_csv(TRAIN_FILE)
    print(training_set.describe())


if __name__ == '__main__':
    main()
