#%%
from copy import deepcopy
import glob
import math
import os
import pickle
import re
import numpy as np
import statistics
from collections import Counter

# Getting data from logs in different formats for plotting

# Get History Auto-Sklearn
def getPerformanceHistory_autosklearn(dataName : str, total=False, operatorsInBestPipeline=False) -> dict:
    performanceHistory_autosklearn = {}

    autosklearnRunPath = f"autosklearn_own/runs/{dataName}"

    for modelFolder in [f for f in os.listdir(autosklearnRunPath) if not f.startswith('.')]:
        modelName = modelFolder.split('Own')[1]
        for runID in [f for f in os.listdir(f"{autosklearnRunPath}/{modelFolder}") if not f.startswith('.')]:
            if operatorsInBestPipeline:
                filepath = glob.glob(f"{autosklearnRunPath}/{modelFolder}/{runID}/*:*.log")[0]
                with open(filepath, "r") as myfile:
                    pipelineFile = myfile.read()
                    pipelineString = \
                    re.findall(r"(?s)Final Incumbent: Configuration:(.*?)\[WARNING\]", pipelineFile, re.MULTILINE)[0]

                    pipelineOperators = re.findall(r"__choice__,\sValue:\s'(.*)'", pipelineString)
                    pipelineOperators = list(dict.fromkeys(pipelineOperators))

                    # keep only feature engineering operators
                    notFeatureEngineeringOperators = ["OwnLinearReg", "OwnLassoLarsCV", "OwnDecisionTree",
                                                      "OwnRandomForest", "make_union", "none", "no_encoding"]
                    pipelineOperators = [elem for elem in pipelineOperators if elem not in notFeatureEngineeringOperators]
                    performanceHistory_autosklearn.setdefault(modelName, []).extend(pipelineOperators)
            else:
                filepath = glob.glob(f"{autosklearnRunPath}/{modelFolder}/{runID}/performanceHistory*.pkl")[0]
                # hidden folders
                if total:
                    score = filepath.split("_score")[1]
                    score = int(score[:score.find(".")])
                    performanceHistory_autosklearn.setdefault(modelName, []).append(score)
                else:
                    with open(filepath, 'rb') as file:
                        performanceHistory_autosklearn.setdefault(modelName,[]).append(pickle.load(file))

    return performanceHistory_autosklearn


# Get History TPOT
def getPerformanceHistory_TPOT(dataName : str, byGeneration=False, total=False, operatorsInBestPipeline=False, returnCount=False, verbosity=False) -> dict:
    # for finding the number of operators for best pipeline
    def getBestCVScore(filepath):
        with open(filepath, "r") as myfile:
            pipelineFile = myfile.read()
            m = re.findall("-\d+.\d+", pipelineFile, re.IGNORECASE)[0]
            m = float(m)
            return m

    performanceHistory_TPOT = {}

    # Count for all individuals
    operatorCount = []
    nindividualsCount = []
    tmpOperatorCount = []
    nOperatorsInBest = []
    # measure per run/model
    operatorsHistory = {}
    nindividualsHistory = {}

    # Check occurance of different operators
    nHighDOFFeatureSelectors = 0
    nHighDOFNystroem = 0
    nLowDOFOperators = 0
    nInvalidPipelines = 0
    nIndividualsInGeneration = 0

    tpotRunPath = f"tpot_own/runs/{dataName}"

    for modelFolder in [f for f in os.listdir(tpotRunPath) if not f.startswith('.')]:
        for runID in [f for f in os.listdir(f"{tpotRunPath}/{modelFolder}") if not f.startswith('.')]:
            modelName = modelFolder.split('_')[0]
            filepath = glob.glob(f"{tpotRunPath}/{modelFolder}/{runID}/time*.py")[0]
            if operatorsInBestPipeline:
                # get list of operators in best pipeline
                with open(filepath, "r") as myfile:
                    pipelineFile = myfile.read()
                    pipelineString = \
                    re.findall(r"(?s)make_pipeline(\(.+)exported_pipeline.fit", pipelineFile, re.MULTILINE)[0]
                    pipelineOperators = re.findall(r"\s+(.*?)\(", pipelineString)
                    # Remove dublicates
                    pipelineOperators = list(dict.fromkeys(pipelineOperators))
                    # keep only feature engineering operators
                    notFeatureEngineeringOperators = ["LinearRegression", "LassoLarsCV", "DecisionTreeRegressor", "RandomForestRegressor", "make_union", "make_pipeline", "FunctionTransformer"]
                    pipelineOperators = [elem for elem in pipelineOperators if elem not in notFeatureEngineeringOperators]
                    performanceHistory_TPOT.setdefault(modelName, []).extend(pipelineOperators)
            elif total:
                score = filepath.split("_score")[1]
                score = int(score[:score.find("_")])
                performanceHistory_TPOT.setdefault(modelName, []).append(score)
            else:
                with open(f"{tpotRunPath}/{modelFolder}/{runID}/performance_history.pkl", 'rb') as file:
                    performDict = pickle.load(file)
                    optimisatioHist = []
                    operatorHist = []
                    individualsHist = []
                    bestInGeneration = float("inf")
                    currentGeneration = 0
                    for key, value in performDict.items():
                        # operators in best pipeline
                        cvScoreBestPipeline = getBestCVScore(filepath)
                        if value["internal_cv_score"] == cvScoreBestPipeline:
                            nOperatorsInBest.append(value["operator_count"])

                        # 5000 for invalid pipelines
                        if not (value["operator_count"] == 5000):
                            # occurance of operators
                            operatorCount.append(value["operator_count"])
                            tmpOperatorCount.append(value["operator_count"])

                        #if value["operator_count"] == 2 and "CombineDFs" not in key:
                        if True:
                            # contains operator with many DOF
                            if any(dofOperator in key for dofOperator in ["SelectPercentile", "SelectFromModel", "SelectFwe"]):
                                nHighDOFFeatureSelectors = nHighDOFFeatureSelectors + 1
                            elif "Nystroem" in key:
                                nHighDOFNystroem = nHighDOFNystroem + 1
                            else:
                                nLowDOFOperators = nLowDOFOperators + 1
                                #print(key)

                        # Measures individual fitness
                        runPerf = value["internal_cv_score"]
                        # Remove invalid configurations (due to time, error, etc)
                        # nan if operator is used in a way that make prediction unavailable, e.g. reduces feature size to 0
                        if math.isnan(runPerf) or math.isinf(runPerf):
                            nInvalidPipelines += 1
                            runPerf = float("inf")
                        else:
                            runPerf = (-int(runPerf))

                        # get performance history per model or generation
                        if byGeneration:
                            if value["generation"] == currentGeneration:
                                nIndividualsInGeneration += 1
                                if runPerf <= bestInGeneration:
                                    bestInGeneration = runPerf
                            else:
                                # collecting meta informatino
                                if nIndividualsInGeneration >= 50:
                                    # There are cases with more then 50 individuals in generation.
                                    # Bug when time constraint hits, very vew cases
                                    #print(f"Limit of 50 individuals, not followed with: {nIndividualsInGeneration}")
                                    nIndividualsInGeneration = 49

                                nindividualsCount.append(nIndividualsInGeneration)
                                operatorHist.append(np.median(tmpOperatorCount))
                                individualsHist.append(nIndividualsInGeneration)
                                tmpOperatorCount = []

                                #print(f"In generation {currentGeneration} were {nIndividualsInGeneration}")
                                nIndividualsInGeneration = 0
                                optimisatioHist.append(bestInGeneration)
                                currentGeneration = value["generation"]
                                bestInGeneration = runPerf
                        else:
                            optimisatioHist.append(runPerf)

                    performanceHistory_TPOT.setdefault(modelName,[]).append(optimisatioHist)
                    operatorsHistory.setdefault(modelName,[]).append(operatorHist)
                    nindividualsHistory.setdefault(modelName,[]).append(individualsHist)

    if (not total) and verbosity:
        print("number of operators with low num. of param." + str(nLowDOFOperators))
        print("Pecent of occurance Nystroem " + str(nHighDOFNystroem / (nLowDOFOperators + nHighDOFNystroem + nHighDOFFeatureSelectors)))
        print("number of feature selectors with high num. of param." + str(nHighDOFFeatureSelectors))
        print("Percentage operators low DOF: " + str(nLowDOFOperators/(nHighDOFNystroem + nHighDOFFeatureSelectors)))
        print("number of invalid pipelines " + str(nInvalidPipelines))
        print()
    if returnCount:
        return operatorsHistory, operatorCount, nindividualsHistory, nindividualsCount, nOperatorsInBest
    return performanceHistory_TPOT


def getPerformances_autofeat(dataName : str, total=False, inPercent=None):
    # autofeat has additional step parameter for synthetic testing
    dataName=f"{dataName}_"
    # could not do 3rd engineering step
    maxFeatEngSteps = 3
    if "rossmann" in dataName:
        maxFeatEngSteps = 2

    maxSelSteps = 5

    #if "taxi" in dataName:
    #    maxFeatEngSteps = 2

    performanceTotal = {}
    featureCountTotal = {}
    dataStructure = [[[] for i in range(maxSelSteps)] for j in range(maxFeatEngSteps)]

    autofeatRunPath = f"autofeat_own/runs/{dataName}"
    for runID in [f for f in os.listdir(f"{autofeatRunPath}") if not f.startswith('.')]:
        for fileName in [f for f in os.listdir(f"{autofeatRunPath}/{runID}") if not f.startswith('.') and f.__contains__("_performance")]:
            engineeringSteps = int(re.findall(r'feng(\d+)', fileName)[0]) - 1
            selectionSteps = int(re.findall(r'fsel(\d+)', fileName)[0]) - 1
            with open(f"{autofeatRunPath}/{runID}/{fileName}", 'rb') as file:
                fileContent = pickle.load(file)
                for key, value in fileContent.items():
                    if key =="new_features":
                        continue
                    performanceRMSE = np.sqrt(value)
                    #print("RMSE "+ str(int(performanceRMSE)) + "  -   " + str(inPercent[key]))
                    if inPercent is not None:
                        performanceRMSE = (inPercent[key] - performanceRMSE) / inPercent[key] * 100
                    #print(int(performanceRMSE))
                    performanceTotal.setdefault(key, deepcopy(dataStructure))[engineeringSteps][selectionSteps].append(performanceRMSE)
                    featureCountTotal.setdefault(key, deepcopy(dataStructure))[engineeringSteps][selectionSteps].append(len(fileContent["new_features"]))

    performanceOverSettings = {}
    for key in performanceTotal.keys():
        performanceOfModel = np.array(deepcopy(performanceTotal[key]))
        if total:
            for i in range(len(performanceOfModel[0,0,:])):
                if inPercent is not None:
                    performanceOverSettings.setdefault(key, []).append(np.max(performanceOfModel[:,:,i]))
                else:
                    performanceOverSettings.setdefault(key, []).append(np.min(performanceOfModel[:,:,i]))
        else:
            numberOfFeatures = np.array(deepcopy(featureCountTotal[key]))
            for featEngSteps in range(maxFeatEngSteps):
                for featSelSteps in range(maxSelSteps):
                    performanceTotal[key][featEngSteps][featSelSteps] = np.median(performanceOfModel[featEngSteps, featSelSteps, :])
                    featureCountTotal[key][featEngSteps][featSelSteps] = np.median(numberOfFeatures[featEngSteps, featSelSteps, :])
    if total:
        return performanceOverSettings
    else:
        return [performanceTotal, featureCountTotal]


def getTotalPerformances(dataName : str, default: dict):
    totalPerformance = {}
    totalPerformance["Auto-sklearn"] = getPerformanceHistory_autosklearn(dataName, total=True)
    totalPerformance["TPOT"] = getPerformanceHistory_TPOT(dataName, total=True)
    totalPerformance["Autofeat"] = getPerformances_autofeat(dataName, total=True)
    totalPerformance.update(default)
    return totalPerformance

# List types of operators
def operatorsUsed(tpot=True):
    if tpot:
        operatorCount_s = getPerformanceHistory_TPOT("synthetic_0.7", operatorsInBestPipeline=True)
        operatorCount_r = getPerformanceHistory_TPOT("rossmann_0.7", operatorsInBestPipeline=True)
        operatorCount_t = getPerformanceHistory_TPOT("taxitrip_0.7", operatorsInBestPipeline=True)
        modelOrder = ["RandomForestRegressor", "DecisionTreeRegressor", "LinearRegression", "LassoLarsCV"]
    else:
        operatorCount_s = getPerformanceHistory_autosklearn("synthetic_0.7", operatorsInBestPipeline=True)
        operatorCount_r = getPerformanceHistory_autosklearn("rossmann_0.7", operatorsInBestPipeline=True)
        operatorCount_t = getPerformanceHistory_autosklearn("taxitrip_0.7", operatorsInBestPipeline=True)
        modelOrder = ["RandomForest", "DecisionTree", "LinearReg", "LassoLarsCV"]

    pipelineOperators = []
    for dataSets in [operatorCount_r, operatorCount_t, operatorCount_s]:
        for key, value in dataSets.items():
            pipelineOperators.extend(list(dict.fromkeys(value)))
    pipelineOperators = list(dict.fromkeys(pipelineOperators))
    list.sort(pipelineOperators)


    operatorMatrix = []
    for dataSet in [operatorCount_s, operatorCount_r, operatorCount_t, ]:
        for key in modelOrder:
            operatorVector = []
            counterOperators = Counter(dataSet[key])
            for operator in pipelineOperators:
                operatorVector.append(counterOperators.get(operator, 0))
            operatorMatrix.append(operatorVector)
    # Transpose lists
    operatorMatrix = list(map(list, zip(*operatorMatrix)))
    return operatorMatrix, pipelineOperators

def getMedianPerformance(dataName="rossmann_0.7", framework="TPOT"):
    #% Get performance of median pipeline
    if framework == "TPOT":
        perHistory = getPerformanceHistory_TPOT(dataName, total=True)
    elif framework == "autosklearn":
        perHistory = getPerformanceHistory_autosklearn(dataName, total=True)
    elif framework == "autofeat":
        perHistory = getPerformances_autofeat(dataName, total=True)

    if "LinearReg" in perHistory:
        perHistory["LinearRegression"] = perHistory.pop("LinearReg")
    if "DecisionTreeRegressor" in perHistory:
        perHistory["DecisionTree"] = perHistory.pop("DecisionTreeRegressor")
        perHistory["RandomForest"] = perHistory.pop("RandomForestRegressor")

    medianPerfDict = []
    for key in ["RandomForest", "DecisionTree", "LinearRegression", "LassoLarsCV"]:
        medianPerf = statistics.median_low(perHistory[key])
        print(f"for {key}: {medianPerf}")
        medianPerfDict.append(medianPerf)
    return medianPerfDict