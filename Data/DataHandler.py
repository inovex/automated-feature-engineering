import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# !Data is missing, needs to be downloaded from the cited sources.

# Numerical, categorical and date types, intended for supervised regression
def getData_rossmann(preprocessed=False, allowDate=True, trainSize=0.1, specifics="None", nDataPoints=None, **kwargs):
    data_train = pd.read_csv(filePath('rossmann-store-sales/train.csv'), low_memory=False, parse_dates=[2])
    data_extra_features = pd.read_csv(filePath('rossmann-store-sales/store.csv'), low_memory=False)
    # test = pandas.read_csv('./Data/rossmann-store-sales/test.csv')

    # Merging extra data
    data_train = pd.merge(data_train, data_extra_features, how="inner", on="Store")

    # for description of features as needed in autosklearn
    if preprocessed:
        # Q1 = data_train.Sales.quantile(0.25)
        # Q3 = data_train.Sales.quantile(0.75)
        # IQR = Q3 - Q1
        # outlier = ~((data_train.Sales < (Q1 - 1.5 * IQR)) | (data_train.Sales > (Q3 + 1.5 * IQR)))
        # print(outlier.value_counts())
        # data_train = data_train[outlier]

        if allowDate:
            data_train['Year'] = data_train.Date.dt.year
            data_train['Month'] = data_train.Date.dt.month
            data_train['Day'] = data_train.Date.dt.day
            data_train['WeekOfYear'] = data_train.Date.dt.weekofyear
        data_train.drop("Date", 1, inplace=True)

        for objectName in data_train.select_dtypes(object).columns.values:
            # All categorical values in numerical, most simple transformation, additional one-hot in tpot
            data_train[objectName] = data_train[objectName].astype('category').cat.codes

        if specifics == "BENCHMARK" or specifics == "AUTOFEAT":
            # Missing Values: fill NaN with a median value
            data_train['CompetitionDistance'].fillna(0, inplace=True)
            data_train.fillna(0, inplace=True)

        if specifics == "AUTOSKLEARN":
            # only does one-hot enconding on cateogorical features in numeric format, ignored for store
            feature_types = (['numerical']*3 + ['categorical'] * 6 + ['numerical'] * 3
                             + ['categorical'] + ['numerical'] * 2 + ['categorical'] + ['numerical'] * 4)

        # Autofeat one-hot enciding, may nor may not be used in engineering
        elif specifics == "AUTOFEAT":
            feature_types = ["Open", "Promo", "StateHoliday", "SchoolHoliday", "StoreType",
                               "Assortment", "Promo2", "PromoInterval"]
            # Bug in autofeat V:1.1.1 with one-hot encoding naming (-1,1 as absolute in name)
            data_train.PromoInterval = data_train.PromoInterval.replace(to_replace=-1, value=3)

    if nDataPoints is not None:
        # Floating point issue
        if nDataPoints % 10 == 0:
            nDataPoints = nDataPoints-1
        data_train = data_train.sample(n=nDataPoints, random_state=1)

    target = data_train.pop("Sales")
    X_train, X_test, y_train, y_test = train_test_split(data_train, target.values, train_size=trainSize,
                                                        test_size=(1-trainSize), random_state=42)
    X_train = pd.DataFrame(X_train, columns=data_train.columns)
    X_test = pd.DataFrame(X_test, columns=data_train.columns)
    #print(data_train.columns)

    if 'feature_types' in locals():
        return [X_train, X_test, y_train, y_test, feature_types]
    else:
        return [X_train, X_test, y_train, y_test]


# Mostly numerical and some categorical types, intended for regression, prediction of trip duration
def getData_taxitrip(preprocessed=False, allowDate=True, trainSize=0.1, specifics="None", nDataPoints=None, **kwargs):
    # No missing values etc
    data_train = pd.read_csv(filePath('taxiTrip/train.csv'), low_memory=False,
                             parse_dates=["pickup_datetime", "dropoff_datetime"])

    if preprocessed:
        # removed outliers with interquartile range since they skew mse
        Q1 = data_train.trip_duration.quantile(0.25)
        Q3 = data_train.trip_duration.quantile(0.75)
        IQR = Q3 - Q1
        outlier = ~((data_train.trip_duration < (Q1 - 1.5 * IQR)) | (data_train.trip_duration > (Q3 + 1.5 * IQR)))
        #print(outlier.value_counts())
        data_train = data_train[outlier]

        if allowDate:
            data_train['pickup_weekday'] = data_train.pickup_datetime.dt.weekday
            data_train['pickup_weekOfYear'] = data_train.pickup_datetime.dt.weekofyear
            data_train['pickup_hour'] = data_train.pickup_datetime.dt.hour
            data_train['pickup_minute'] = data_train.pickup_datetime.dt.minute
            data_train['pickup_second'] = data_train.pickup_datetime.dt.second

        # Dropoff feature obviously only in tain, also drop id of route
        data_train.drop(["pickup_datetime", "dropoff_datetime"], 1, inplace=True)

        for objectName in data_train.select_dtypes(object).columns.values:
            # All categorical values in numerical, most simple transformation, additional one-hot in tpot
            data_train[objectName] = data_train[objectName].astype('category').cat.codes

        if specifics == "BENCHMARK" or specifics == "AUTOFEAT":
            data_train.fillna(0, inplace=True)

        # autoskelarn and autofeat always uses onehot aswell, id removed
        if specifics == "AUTOSKLEARN":
            # only does one-hot enconding on cateogorical features in numeric format, ignored for store
            feature_types = (
                        ['numerical'] + ['categorical'] + ['numerical']*5 + ['categorical'] + ['numerical'] * 5)
        elif specifics == "AUTOFEAT":
            feature_types = ["vendor_id", "store_and_fwd_flag"]

    if nDataPoints is not None:
        # Floating point issue
        if nDataPoints % 10 == 0:
            nDataPoints = nDataPoints-1
        data_train = data_train.sample(n=nDataPoints, random_state=1)

    target = data_train.pop("trip_duration")

    X_train, X_test, y_train, y_test = train_test_split(data_train, target.values, train_size=trainSize,
                                                        test_size=(1-trainSize), random_state=42)

    if 'feature_types' in locals():
        return [X_train, X_test, y_train, y_test, feature_types]
    else:
        return [X_train, X_test, y_train, y_test]


def getData_xor(featureTransformed=False, trainSize=0.1, specifics="None", **kwargs):
    np.random.seed(42)
    x1 = np.random.choice(2, 1000)
    np.random.seed(24)
    x2 = np.random.choice(2, 1000)

    target = np.asarray([1 if (x1[i] != x2[i]) else 0 for i in range(len(x1))])

    if not featureTransformed:
        X = np.vstack([x1, x2]).T
        df = pd.DataFrame(X, columns=["x1", "x2"])
    else:
        xnew = x1*x2
        X = np.vstack([x1, x2, xnew]).T
        df = pd.DataFrame(X, columns=["x1", "x2", "xnew"])

    # reset random seed
    np.random.seed()

    X_train, X_test, y_train, y_test = train_test_split(df, target, train_size=trainSize, random_state=42)

    if "AUTOFEAT" in specifics or "AUTOSKLEARN" in specifics:
        return X_train, X_test, y_train, y_test, None
    else:
        return X_train, X_test, y_train, y_test


# Synthetic data
def getData_synthetic(preprocessed=False, trainSize=0.1, specifics="None", steps="", **kwargs):
    np.random.seed(42)

    nDataPoints = 10000

    # random feature sample from uniform/standard distribution
    x1 = np.random.rand(nDataPoints)
    x2 = np.random.randn(nDataPoints)
    x3 = np.random.rand(nDataPoints)
    x4 = np.random.randn(nDataPoints)

    x1 = np.interp(x1, (0, 1), (1, 100))
    x2 = np.interp(x2, (-1, 1), (-100, 100))
    x3 = np.interp(x3, (0, 1), (1, 100))
    x4 = np.interp(x4, (-1, 1), (-100, 100))

    # Target function to be approximated
    target = 2 + 15 * x1 + 3 / (x2 - 1 / x3) + 5 * (x2 + np.log(x1)) ** 2 - x4**4

    features = [x1, x2, x3, x4]
    columNames = ["x1", "x2", "x3", "x4"]

    if preprocessed:
        # "categorical" feature with 3 values and different effects,
        # would be expected to be seperated
        # effect same as in cat1
        if "cat0" in steps:
            xCat = np.random.randint(0,3,nDataPoints)

            i = 0
            while i < len(target):
                if xCat[i] == 0:
                    target[i] = 0
                elif xCat[i] == 1:
                    target[i] = target[i]**2
                i += 1
            features.append(xCat)
            columNames.append("xCat")

        if "cat1" in steps:
            xCat = np.random.randint(0,2,nDataPoints)

            i = 0
            while i < len(target):
                if xCat[i] == 1:
                    target[i] = target[i]*5
                i += 1

            features.append(xCat)
            columNames.append("xCat")

        if "cat2" in steps:
            xCat = np.random.randint(0,2,nDataPoints)

            tmpX4 = x4.copy()
            i = 0
            while i < len(x4):
                if xCat[i] == 1:
                    tmpX4[i] = 0
                i += 1

            target = 2 + 15 * x1 + 3 / (x2 - 1 / x3) + 5 * (x2 + np.log(x1)) ** 2 - tmpX4 ** 4

            features.append(xCat)
            columNames.append("xCat")

        # Introduce 5% of standard deviation as noise
        if "targetNoise" in steps:
            target = target + 0.05 * target.std() * np.random.randn(nDataPoints)

        # Introduce 5% outliers to a feature, should be removed or adjusted
        if "featureNoise" in steps:
            percentage = 0.1
            x4 = x4 + percentage * x4.std() * np.random.randn(nDataPoints)
            features[3] = x4

        # Lose feature
        if "featureLoss" in steps:
            del(features[3])
            del(columNames[3])

        # Introduce feature twice
        if "featureTwice" in steps:
            features.append(x1)
            columNames.append("x1copy")

        # Add random unuseful features
        if "featureRandom" in steps:
            meaningless1 = np.random.rand(nDataPoints)
            meaningless2 = np.random.randn(nDataPoints)
            meaningless1 = np.interp(meaningless1, (0, 1), (1, 100))
            meaningless2 = np.interp(meaningless2, (-1, 1), (-100, 100))
            meaningless3 = list(range(1, nDataPoints+1))
            features.extend([meaningless1, meaningless2, meaningless3])
            columNames.extend(["meaningless1", "meaningless2", "meaningless3"])

    # reset random seed
    np.random.seed()

    # Finish dataset
    X = np.vstack(features).T
    df = pd.DataFrame(X, columns=columNames)

    X_train, X_test, y_train, y_test = train_test_split(df, target, train_size=trainSize, random_state=42)

    feature_types = None
    if preprocessed:
        if specifics == "AUTOSKLEARN" and "cat" in steps:
            # only does one-hot enconding on cateogorical features in numeric format, ignored for store
            feature_types = (['numerical'] * 4 + ['categorical'])
        elif specifics == "AUTOSKLEARN" and "featureRandom" in steps:
            feature_types = (['numerical'] * 4 + ['numerical'] * 3)
        elif specifics == "AUTOSKLEARN":
            feature_types = (['numerical'] * 4)
        elif specifics == "AUTOFEAT" and "cat" in steps:
            feature_types = ["xCat"]

    if "AUTOFEAT" in specifics or "AUTOSKLEARN" in specifics:
            return X_train, X_test, y_train, y_test, feature_types
    else:
        return X_train, X_test, y_train, y_test
#%%

def filePath(file):
    if os.path.dirname(__file__) == '':
        return os.path.join('Data', file)
    return os.path.join(os.path.dirname(__file__), file)


def stringToMethod(dataName):
    if "taxi" in dataName:
        return getData_taxitrip
    elif "rossmann" in dataName:
        return getData_rossmann
    elif "synthetic" in dataName:
        return getData_synthetic


def createDictPath(path, folderID, createPath=True):
    if folderID is None:
        if not os.path.exists(path):
            path = f'{path}/1'
        else:
            evaluationNumber = len(os.listdir(path))+1
            path = f'{path }/{evaluationNumber}'
    else:
        path = f'{path}/{folderID}'

    if createPath:
        os.makedirs(path, exist_ok=True)
    return path