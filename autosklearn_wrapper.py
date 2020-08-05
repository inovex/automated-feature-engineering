import argparse
import os
import pickle
import shutil
import time
from math import sqrt

import autosklearn.classification
import autosklearn.regression
import numpy as np
from autosklearn.pipeline.components.base import \
    AutoSklearnRegressionAlgorithm
from joblib import dump
from sklearn.metrics import mean_squared_error

from Data import DataHandler
from autosklearn_configuration import OwnDecisionTree, OwnRandomForest, OwnLinearReg, OwnLassoLarsCV


def rmse_scorer(y_true, y_pred):
    # rmse only from 0.22 onwards in scikit learn, autosklearn uses currently 0.21
    return sqrt(mean_squared_error(y_true, y_pred))

class AutosklearnWrapper:
    def __init__(self, model, dataFunction, trainSize=0.7, preprocessed=True, folderID=None, nDataPoints=100000):
        if model in "DecisionTree":
            self.model = OwnDecisionTree
        elif model in "RandomForest":
            self.model = OwnRandomForest
        elif model in "LinearRegression":
            self.model = OwnLinearReg
        elif model in "LassoLarsCV":
            self.model = OwnLassoLarsCV
        else:
            print(f'{model} not found!')
            exit(1)

        self.nDataPoints = nDataPoints
        self.preprocessed = preprocessed
        self.trainSize = trainSize
        self.modelName = self.model.__name__
        self.dataFunction = dataFunction
        self.dataUsedName = self.dataFunction.__name__.split("_")[1]

        # Export path
        self.savePath = f'{os.path.dirname(os.path.abspath(__file__))}/runs/{self.dataUsedName}_{self.trainSize}/{self.modelName}'
        # Add folder with number that represents evaluation run
        self.savePath = DataHandler.createDictPath(self.savePath, folderID, False)

    def regression(self, timeMax=60):
        # Add regression component
        autosklearn.pipeline.components.regression.add_regressor(self.model)

        # Get data
        X_train, X_test, y_train, y_test, feature_types = self.dataFunction(preprocessed=self.preprocessed, specifics="AUTOSKLEARN",
                                                                            trainSize=self.trainSize, nDataPoints=self.nDataPoints)
        rmse_metric = autosklearn.metrics.make_scorer(
            name="rmse",
            score_func=rmse_scorer,
            optimum=0,
            greater_is_better=False,
            needs_proba=False,
            needs_threshold=False,
        )

        # measure runtime
        start_time = time.time()

        # autosklearn automated feature engineering
        automl = autosklearn.regression.AutoSklearnRegressor(
            seed=np.random.randint(0, 100000),
            ensemble_size=0,
            time_left_for_this_task=timeMax*60,
            per_run_time_limit=30*60,
            tmp_folder=self.savePath,
            delete_tmp_folder_after_terminate=False,
            output_folder=f'{self.savePath}_output',
            delete_output_folder_after_terminate=True,
            include_estimators=[self.modelName, ],
            # Does not take autosklearn and smac resources into account
            ml_memory_limit=7000,
            # cv for comparability
            resampling_strategy='cv',
            resampling_strategy_arguments={'folds': 4},
            # Timeout hits earlier
            smac_scenario_args={'runcount_limit': 5000}
        )

        #  Fit changes data in place, need original for refit
        automl.fit(X_train.copy(), y_train.copy(), dataset_name=self.dataUsedName, feat_type=feature_types, metric=rmse_metric)
        print(f"Fit time {int(divmod(time.time() - start_time, 60)[0])}")

        # final ensemble on the whole dataset.
        automl.fit_ensemble(y_train.copy(), ensemble_size=1)
        print(f"Fit ensemble time {int(divmod(time.time() - start_time, 60)[0])}")
        automl.refit(X_train.copy(), y_train.copy())
        print(f"Refit time {int(divmod(time.time() - start_time, 60)[0])}")

        total_time = int(divmod(time.time() - start_time, 60)[0])

        # export model
        dump(automl, f"{self.savePath}/model_time{total_time}.joblib")

        # make predictions
        predictions = automl.predict(X_test)
        predictionMetric = rmse_scorer(y_test, predictions)

        # Export predictions and further stats
        with open(f"{self.savePath}/performanceHistory_score{predictionMetric}.pkl", 'wb') as file:
            pickle.dump(automl.cv_results_["mean_test_score"], file)
        with open(f"{self.savePath}/sprintStatistics.pkl", 'wb') as file:
            pickle.dump(automl.sprint_statistics(), file)

        # Remove large model files
        shutil.rmtree(f'{self.savePath}/.auto-sklearn', ignore_errors=True)


# command-line for ease of use
parser = argparse.ArgumentParser(description='Autosklearn input parser')
parser.add_argument('--time', type=int, help='Time for the optimisation in minutes', default=1)
parser.add_argument('--model', help='Name of class chosen for evaluation')
parser.add_argument('--data', help='Name of data')
parser.add_argument('--trainSize', help='Train size', default=0.7, type=float)
parser.add_argument('--problem', help='Regression or classification problem')
parser.add_argument('--folderID', help='ID for folder')
parser.add_argument('--nDataPoints', type=int, help='Reduce data to subsample size.', default=100000)
args = parser.parse_args()

if "reg" in args.problem:
    AutosklearnWrapper(model=args.model, dataFunction=DataHandler.stringToMethod(args.data), trainSize=args.trainSize,
                       folderID=args.folderID, nDataPoints=args.nDataPoints).regression(args.time)
else:
    print("Not supported")

