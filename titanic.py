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

# Models and metrics
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

import pipelines as pl

TRAIN_FILE = r'inputs/train.csv'
TEST_FILE = r'inputs/test.csv'

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
    models = [
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

    # Find the best model
    accuracy = 0
    best_model = None
    for index, model in enumerate(models):
        result, m = run_grid_search_cv(train_x, train_y.values, *model, params[index])
        if result > accuracy:
            accuracy = result
            best_model = m

    predict(best_model)

if __name__ == '__main__':
    main()
