# Extending default models

from ConfigSpace.configuration_space import ConfigurationSpace

from autosklearn.pipeline.components.base import (
    AutoSklearnRegressionAlgorithm,
)
from autosklearn.pipeline.constants import *


class OwnDecisionTree(AutoSklearnRegressionAlgorithm):
    def __init__(self, random_state=None):
        self.random_state = random_state
        self.estimator = None

    def fit(self, X, y, sample_weight=None):
        from sklearn.tree import DecisionTreeRegressor

        self.estimator = DecisionTreeRegressor(random_state=self.random_state)
        self.estimator.fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError
        return self.estimator.predict(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'DT',
                'name': 'Decision Tree Classifier',
                'handles_regression': True,
                'handles_classification': False,
                'handles_multiclass': False,
                'handles_multilabel': False,
                'is_deterministic': False,
                'input': (DENSE, SPARSE, UNSIGNED_DATA),
                'output': (PREDICTIONS,)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()
        return cs


class OwnRandomForest(AutoSklearnRegressionAlgorithm,):
    def __init__(self, random_state=None):
        self.estimator = None
        self.random_state = random_state

    def fit(self, X, y):
        from sklearn.ensemble import RandomForestRegressor
        # To match configuration of others
        self.estimator = RandomForestRegressor(random_state=self.random_state, n_estimators=10)
        self.estimator.fit(X, y)
        return self

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError
        return self.estimator.predict(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'RF',
                'name': 'Random Forest Regressor',
                'handles_regression': True,
                'handles_classification': False,
                'handles_multiclass': False,
                'handles_multilabel': False,
                'is_deterministic': True,
                'input': (DENSE, SPARSE, UNSIGNED_DATA),
                'output': (PREDICTIONS,)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()
        return cs


class OwnLinearReg(AutoSklearnRegressionAlgorithm,):
    def __init__(self, random_state=None):
        self.estimator = None

    def fit(self, X, y):
        from sklearn.linear_model import LinearRegression
        self.estimator = LinearRegression()
        self.estimator.fit(X, y)
        return self

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError
        return self.estimator.predict(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'LR',
                'name': 'Linear Regression',
                'handles_regression': True,
                'handles_classification': False,
                'handles_multiclass': False,
                'handles_multilabel': False,
                'is_deterministic': True,
                'input': (DENSE, SPARSE, UNSIGNED_DATA),
                'output': (PREDICTIONS,)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()
        return cs


class OwnLassoLarsCV(AutoSklearnRegressionAlgorithm,):
    def __init__(self, random_state=None):
        self.estimator = None

    def fit(self, X, y):
        from sklearn.linear_model import LassoLarsCV
        self.estimator = LassoLarsCV(cv=5)
        self.estimator.fit(X, y)
        return self

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError
        return self.estimator.predict(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'LL',
                'name': 'LassoLarsCV',
                'handles_regression': True,
                'handles_classification': False,
                'handles_multiclass': False,
                'handles_multilabel': False,
                'is_deterministic': True,
                'input': (DENSE, SPARSE, UNSIGNED_DATA),
                'output': (PREDICTIONS,)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()
        return cs