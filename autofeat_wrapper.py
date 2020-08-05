import argparse
import os
import pickle
import sys
import time

from autofeat import AutoFeatRegressor
from joblib import dump
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoLarsCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor

from Data import DataHandler


class AutofeatWrapper:
    def __init__(self, dataFunction, trainSize=0.1, feateng_steps=3, featuresel_steps=5, steps="", folderID=None, nDataPoints=100000):
        self.trainSize = trainSize
        self.dataFunction = dataFunction
        self.dataUsedName = self.dataFunction.__name__.split("_")[1]
        self.feateng_steps = feateng_steps
        self.featuresel_steps = featuresel_steps
        self.steps=steps
        self.nDataPoints = nDataPoints
        # Export path
        self.savePath = f'{os.path.dirname(os.path.abspath(__file__))}/runs/{self.dataUsedName}_{self.trainSize}_{steps}'
        # Add folder with number that represents evaluation run
        self.savePath = DataHandler.createDictPath(self.savePath, folderID)

    def regression(self):
        X_train, X_test, y_train, y_test, categoricalCols = \
            self.dataFunction(preprocessed=True, specifics="AUTOFEAT", trainSize=self.trainSize, steps=self.steps, nDataPoints=self.nDataPoints)

        featureEngColumns = None
        # If feature engineering not wanted for categorical values, uncomment
        # if categoricalCols is not None:
        #    featureEngColumns = X_train.columns.values.tolist()
        #    featureEngColumns = [i for i in featureEngColumns + categoricalCols if i not in featureEngColumns or i not in categoricalCols]

        # Measure runtime
        start_time = time.time()
        print(f"Start time: {start_time}")

        # Automated feature engineering with autofeat
        model = AutoFeatRegressor(verbose=1, feateng_steps=self.feateng_steps, featsel_runs=self.featuresel_steps,
                                   categorical_cols=categoricalCols, feateng_cols=featureEngColumns)

        # Fit model and get transformed dataframe with additional features
        x_train_extended = model.fit_transform(X_train, y_train)
        total_time = int(divmod(time.time() - start_time, 60)[0])
        print(f"Time: {total_time}")

        # Export model
        dump(model, f"{self.savePath}/feng{model.feateng_steps}_fsel{model.featsel_runs}_time{total_time}_model.joblib")

        x_test_extended = model.transform(X_test)

        # Predictions
        predictions = {}

        predictionModel = DecisionTreeRegressor()
        predictionModel.fit(x_train_extended, y_train)
        predictions["DecisionTree"] = mean_squared_error(y_test, predictionModel.predict(x_test_extended))
        print(f"Final MSE prediction score: {predictions['DecisionTree']}")

        predictionModel = RandomForestRegressor(n_estimators=10)
        predictionModel.fit(x_train_extended, y_train)
        predictions["RandomForest"] = mean_squared_error(y_test, predictionModel.predict(x_test_extended))
        print(f"Final MSE prediction score: {predictions['RandomForest']}")

        predictionModel = LinearRegression()
        predictionModel.fit(x_train_extended, y_train)
        predictions["LinearRegression"] = mean_squared_error(y_test, predictionModel.predict(x_test_extended))
        print(f"Final MSE prediction score: {predictions['LinearRegression']}")

        predictionModel = LassoLarsCV(cv=5)
        predictionModel.fit(x_train_extended, y_train)
        predictions["LassoLarsCV"] = mean_squared_error(y_test, predictionModel.predict(x_test_extended))
        print(f"Final MSE prediction score: {predictions['LassoLarsCV']}")

        # Additionally save transformations steps since not saved in joblib file
        predictions["new_features"] = model.new_feat_cols_

        # Export predictions
        with open(f"{self.savePath}/feng{model.feateng_steps}_fsel{model.featsel_runs}_performance.pkl", 'wb') as file:
            pickle.dump(predictions, file)

        return model


# command-line for ease of use
parser = argparse.ArgumentParser(description='Autofeat input parser')
parser.add_argument('--data', help='Name of data')
parser.add_argument('--trainSize', help='Train size', default=0.7, type=float)
parser.add_argument('--problem', help='Regression or classification problem', default="regression")
parser.add_argument('--feateng_steps', help='Number of feature engineering steps', default=2, type=int)
parser.add_argument('--featsel_runs', help='Number of feature selection steps', default=5, type=int)
parser.add_argument('--folderID', help='ID for folder')
parser.add_argument('--nDataPoints', type=int, help='Reduce data to subsample size.', default=100000)
args = parser.parse_args()

if len(sys.argv) > 1:
    if "reg" in args.problem:
        autofeatTransformer = AutofeatWrapper(dataFunction=DataHandler.stringToMethod(args.data), trainSize=args.trainSize,
                                              feateng_steps=args.feateng_steps, featuresel_steps=args.featsel_runs,
                                              folderID=args.folderID, nDataPoints=args.nDataPoints)
        # User input 100 means try every combination
        if args.feateng_steps == 100 and args.featsel_runs == 100:
            maxEngSteps = 3
            # dataset not feasible for categorical values with 8gb of RAM
            if "rossmann" in args.data:
                maxEngSteps = 2
            for i in range(maxEngSteps):
                for j in range(5):
                    autofeatTransformer.feateng_steps = i + 1
                    autofeatTransformer.featuresel_steps = j + 1
                    autofeatTransformer.regression()
        elif args.featsel_runs == 100:
            for j in range(5):
                autofeatTransformer.featuresel_steps = j + 1
                autofeatTransformer.regression()
        else:
            autofeatTransformer.regression()
    elif "class" in args.problem:
        pass
    else:
        print("Wrong problem choice")

