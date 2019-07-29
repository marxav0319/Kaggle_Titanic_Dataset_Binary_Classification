"""
"""

# Data Cleaning and Transformations
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

NUMERIC_COLUMNS = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
CATEGORICAL_COLUMNS = ['Sex', 'Embarked']
COLUMNS_TO_DROP = ['PassengerId', 'Name', 'Ticket', 'Cabin']

NUMERIC_PIPELINE = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
    ])

NUMERIC_PIPELINE_W_STANDARDIZATION = Pipeline([
      ('imputer', SimpleImputer(strategy='median')),
      ('std_scaler', StandardScaler())
    ])

CATEGORICAL_ONE_HOT_PIPELINE = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('one_hot_encoder', OneHotEncoder())
    ])

CATEGORICAL_ORDINAL_PIPELINE = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('one_hot_encoder', OrdinalEncoder())
    ])

def get_pipeline(standardize=False, one_hot=False):
    """
    """

    data_pipeline = None
    if standardize and one_hot:
        data_pipeline = ColumnTransformer([
                ('drop_unneeded', 'drop', COLUMNS_TO_DROP),
                ('numeric_pipeline', NUMERIC_PIPELINE_W_STANDARDIZATION, NUMERIC_COLUMNS),
                ('categorical_pipeline', CATEGORICAL_ONE_HOT_PIPELINE, CATEGORICAL_COLUMNS)
            ])
    elif standardize and not one_hot:
        data_pipeline = ColumnTransformer([
                ('drop_unneeded', 'drop', COLUMNS_TO_DROP),
                ('numeric_pipeline', NUMERIC_PIPELINE_W_STANDARDIZATION, NUMERIC_COLUMNS),
                ('categorical_pipeline', CATEGORICAL_ORDINAL_PIPELINE, CATEGORICAL_COLUMNS)
            ])
    elif not standardize and one_hot:
        data_pipeline = ColumnTransformer([
                ('drop_unneeded', 'drop', COLUMNS_TO_DROP),
                ('numeric_pipeline', NUMERIC_PIPELINE, NUMERIC_COLUMNS),
                ('categorical_pipeline', CATEGORICAL_ONE_HOT_PIPELINE, CATEGORICAL_COLUMNS)
            ])
    else:
        data_pipeline = ColumnTransformer([
                ('drop_unneeded', 'drop', COLUMNS_TO_DROP),
                ('numeric_pipeline', NUMERIC_PIPELINE, NUMERIC_COLUMNS),
                ('categorical_pipeline', CATEGORICAL_ORDINAL_PIPELINE, CATEGORICAL_COLUMNS)
            ])

    return data_pipeline
