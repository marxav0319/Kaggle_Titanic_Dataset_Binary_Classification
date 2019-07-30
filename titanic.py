"""
titanic.py

Not sure what I'll do here exactly yet, I need to run an exploratory analysis, feature engineer, and
build models.  The last two I might do often more than once, so I'll need to figure out a good way
to structure this project.

Author: Mark Xavier
"""

import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

# Models and metrics
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score

import pipelines as pl

TRAIN_FILE = r'inputs/train.csv'
TEST_FILE = r'inputs/test.csv'
ACCURACY_LIM = 0.8

def run_grid_search_cv(x, y, model_instance, standardize, one_hot, label, params, verbose=False):
    """
    """

    pipeline = pl.get_pipeline(standardize=standardize, one_hot=one_hot)
    x_prime = pipeline.fit_transform(x)
    clf = GridSearchCV(model_instance, params, cv=5, scoring='accuracy')
    clf.fit(x_prime, y)
    ypred = clf.predict(x_prime)
    accuracy = accuracy_score(y, ypred)
    if verbose:
        print('%s Accuracy: %.3f' % (label, accuracy))
        print(clf.best_estimator_)
        print('\n\n')

    return accuracy, clf.best_estimator_

def get_inputs():
    """
    """

    training_set = pd.read_csv(TRAIN_FILE)
    train_x = training_set.drop(['Survived'], axis=1)
    train_y = training_set.loc[:, 'Survived']

    return train_x, train_y

def create_neural_network(input_shape):
    """
    """

    nn = keras.Sequential()
    nn.add(keras.layers.Input(input_shape))
    nn.add(keras.layers.Dense(30, activation='relu'))
    nn.add(keras.layers.BatchNormalization())
    nn.add(keras.layers.Dense(30, activation='relu'))
    nn.add(keras.layers.BatchNormalization())
    nn.add(keras.layers.Dense(1, activation='sigmoid'))
    nn.compile(optimizer='adam', loss='binary_crossentropy', 
               metrics=['accuracy'])

    return nn

def predict(model):
    """
    """

    test = pd.read_csv('inputs/test.csv')
    pids = test.loc[:, 'PassengerId'].values.tolist()
    pipeline = pl.get_pipeline(False, False)
    x_clean = pipeline.fit_transform(test)
    ypred = model.predict(x_clean).tolist()
    print(len(ypred))
    print(len(pids))
    f = open('out.csv', 'w')
    f.write('PassengerId,Survived\n')
    for pid, pred in zip(pids, ypred):
        f.write('%s,%d\n' % (pid, pred))
    f.close()

def main():
    """
    """

    # Read input
    train_x, train_y = get_inputs()
    
    # list of models
    model_details = [
        (LogisticRegression(solver='lbfgs', max_iter=1000), True, True, 'LogisticRegression'),
        (SGDClassifier(), True, True, 'SGDClassifier'),
        (DecisionTreeClassifier(), False, False, 'DecisionTreeClassifier'),
        (RandomForestClassifier(n_estimators=10), False, False, 'RandomForestClassifier'),
    ]

    params = [
        {'C':[0.2, 0.5, 0.7, 1.0]},
        {'penalty':['l2', 'l1'], 'alpha':[.0001, .001, .00001]},
        {'max_depth':[2, 4, 6, 8]},
        {'n_estimators':[2, 5, 10, 12, 15], 'max_depth':[2, 4, 6, 8, 10]}
    ]

    models = []

    # Get best sklearn models
    for index, dtls in enumerate(model_details):
        accuracy, model = run_grid_search_cv(train_x, train_y.values, *dtls, params[index])
        if accuracy >= ACCURACY_LIM:
            models.append((dtls[-1], model))
    print('Models in use: %d' % len(models))

    # Create an ensemble method
    x_clean = pl.get_pipeline(True, True).fit_transform(train_x)
    voting_classifier = VotingClassifier(models)
    voting_classifier.fit(x_clean, train_y.values)
    ypred = voting_classifier.predict(x_clean)
    print('Accuracy: %.3f' % accuracy_score(train_y.values, ypred))


if __name__ == '__main__':
    main()
