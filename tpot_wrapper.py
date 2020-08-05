import argparse
import os
import pickle
import time

from sklearn.metrics import mean_squared_error
from sklearn.metrics.scorer import make_scorer
from tpot import TPOTRegressor

from Data import DataHandler
from tpot_configuration import regressor_config


class TpotWrapper:
    def __init__(self, dataFunction, generations=100, popSize=100, trainSize=0.7, model:str='sklearn.tree.DecisionTreeRegressor', preprocessed=True, folderID=None, nDataPoints=100000):
        self.generations = generations
        self.popSize = popSize
        self.trainSize = trainSize
        self.model = model
        self.preprocessed = preprocessed
        self.nDataPoints = nDataPoints

        # For ease of use
        if model in "DecisionTree":
            self.model = 'sklearn.tree.DecisionTreeRegressor'
        elif model in "RandomForest":
            self.model = {"sklearn.ensemble.RandomForestRegressor": {'n_estimators': [10]}}
        elif model in "LinearRegression":
            self.model = "sklearn.linear_model.LinearRegression"
        elif model in "LassoLarsCV":
            #Settings from autofeat
            self.model = {"sklearn.linear_model.LassoLarsCV": {'cv': [5]}}

        if isinstance(self.model, str):
            self.model = {self.model: {}}

        self.modelName = list(self.model.keys())[0].split(".")[-1]
        self.dataFunction = dataFunction
        self.dataUsedName = self.dataFunction.__name__.split("_")[1]

        # Export path
        self.savePath = f'{os.path.dirname(os.path.abspath(__file__))}/runs/{self.dataUsedName}_{self.trainSize}/{self.modelName}_gen{self.generations}_pop{self.popSize}'
        # Add folder with number that represents evaluation run
        self.savePath = DataHandler.createDictPath(self.savePath, folderID)

    def regression(self, timeMax=60):
        def rmse_scorer(y_true, y_pred):
            return mean_squared_error(y_true, y_pred, squared=False)
        my_custom_scorer = make_scorer(rmse_scorer, greater_is_better=False)

        print(f"Starting regression with {self.modelName}")
        X_train, X_test, y_train, y_test = self.dataFunction(preprocessed=self.preprocessed, specifics="TPOT", trainSize=self.trainSize, nDataPoints=self.nDataPoints)

        # Change dict for prediction model
        config_copy = regressor_config.copy()
        config_copy.update(self.model)

        # TPOT automated feature engineering
        start_time = time.time()
        tpot = TPOTRegressor(generations=self.generations, population_size=self.popSize, verbosity=2,
                             config_dict=config_copy, max_time_mins=timeMax,
                             max_eval_time_mins=30, cv=4, scoring=my_custom_scorer)

        tpot.fit(X_train, y_train)
        total_time = int(divmod(time.time() - start_time, 60)[0])
        print(tpot.evaluated_individuals_)
        print(f"Time: {total_time}")

        # prediction score
        predictionScore = int(-tpot.score(X_test, y_test))
        print(f"Final MSE prediction score: {predictionScore}")

        # Export model
        tpot.export(f'{self.savePath}/time{total_time}_score{predictionScore}_trainSize{self.trainSize}_PIPE.py')
        # Export History
        with open(f'{self.savePath}/performance_history.pkl', "wb") as handle:
            pickle.dump(tpot.evaluated_individuals_, handle)
        # Export pareto front
        with open(f'{self.savePath}/PARETO.pkl', "wb") as handle:
            pickle.dump(tpot.pareto_front_fitted_pipelines_, handle)


# command-line for ease of use
parser = argparse.ArgumentParser(description='TPOT input parser')
parser.add_argument('--time', type=int, help='Time for the optimisation in minutes', default=1)
parser.add_argument('--model', help='Name of class chosen for evaluation')
parser.add_argument('--data', help='Name of data')
parser.add_argument('--problem', help='Regression or classification problem')
parser.add_argument('--popSize', type=int, help='Population Size', default=100)
parser.add_argument('--generations', type=int, help='Generation Size', default=100)
parser.add_argument('--trainSize', type=float, help='Train size', default=0.7)
parser.add_argument('--folderID', help='ID for folder')
parser.add_argument('--nDataPoints', type=int, help='Reduce data to subsample size.', default=100000)
args = parser.parse_args()

tpotModel = TpotWrapper(model=args.model, dataFunction=DataHandler.stringToMethod(args.data), generations=args.generations,
                        popSize=args.popSize, trainSize=args.trainSize, folderID=args.folderID, nDataPoints=args.nDataPoints)
if "reg" in args.problem:
    tpotModel.regression(args.time)
else:
    print("Not supported")

