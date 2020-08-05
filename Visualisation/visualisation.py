# Plots for contribution and blogpost

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import numpy as np
import pandas as pd

from Data import DataHandler
from Visualisation.getData import getPerformanceHistory_autosklearn, getPerformanceHistory_TPOT, \
    getPerformances_autofeat, getTotalPerformances, operatorsUsed, getMedianPerformance

savePath = "~/"
colors = ["#85144b", "#0074D9", "#2ECC40", "#FFDC00"]


# Visualise optimisations
def visualiseBestSoFar(performanceHistory, x_label="", y_label="MSE", smallerIsBetter=True, title="", saveName="",
                       logScale=False, quantiles=True, limits=None, onlyBestScore=True):
    modelOrder = ["RandomForest", "DecisionTree", "LinearRegression", "LassoLarsCV"]

    # calc best so far
    bestSoFar = performanceHistory.copy()
    if onlyBestScore:
        for key, runList in bestSoFar.copy().items():
            i = 0
            for value in runList:
                if smallerIsBetter:
                    currentMin = float("inf")
                else:
                    currentMin = float("-inf")
                for ii in range(len(value)):
                    if value[ii] == 1:
                        value[ii] = currentMin

                    if smallerIsBetter:
                        if value[ii] <= currentMin:
                            currentMin = value[ii]
                    else:
                        if value[ii] >= currentMin:
                            currentMin = value[ii]
                    value[ii] = currentMin

                bestSoFar[key][i] = value
                i += 1

    # evaluation runs have different lengths since we stopped with a time constrain
    def getPercentileUneven(data, percentile, early=5):
        # sort matrix depending on length
        data = np.array(sorted(data, key=len))

        numberOfEvals = [0]
        percentileResult = []

        # Points where a run ends
        for arrays in data:
            numberOfEvals.append(len(arrays))
        numberOfEvals.append(numberOfEvals[-1])

        # stop 4 early to have at least 5 points to choose median from
        for i in range(len(numberOfEvals) - 2 - early):
            dataSubset = []

            for row in data[i:]:
                dataSubset.append(row[numberOfEvals[i]:numberOfEvals[i + 1]])

            dataSubset = np.asarray(dataSubset)
            perc = np.percentile(dataSubset, percentile, axis=0, interpolation="midpoint")
            percentileResult.extend(perc)
            np.delete(data, 0, 0)

        return percentileResult, np.asarray(numberOfEvals[1:(-1-early)])-1

    # plot
    fig, ax = plt.subplots(dpi=200)

    if logScale:
        ax.set_yscale('log')

    # Fix different names in frameworks:
    if "LinearReg" in bestSoFar:
        bestSoFar["LinearRegression"] = bestSoFar.pop("LinearReg")
    if "DecisionTreeRegressor" in bestSoFar:
        bestSoFar["DecisionTree"] = bestSoFar.pop("DecisionTreeRegressor")
        bestSoFar["RandomForest"] = bestSoFar.pop("RandomForestRegressor")

    longestRunPos = 0
    i = 0
    for key in modelOrder:
        elem = bestSoFar[key]

        # calculate the lower and upper percentile groups
        perc1, _ = getPercentileUneven(elem, 25)
        perc2, _ = getPercentileUneven(elem, 75)
        median, cutOff = getPercentileUneven(elem, 50)

        # fill lower and upper percentile groups
        plt.fill_between(range(0, len(perc1)), perc1, perc2, alpha=0.5, color=colors[i], edgecolor=None)
        plt.plot(median, label=f"{key}", color=colors[i])
        plt.scatter(cutOff, np.asarray(median)[cutOff], marker="|", color='r', label="Run Completed" if i == 0 else "", alpha=0.5)
        if quantiles:
            for runPerf in elem:
                plt.plot(runPerf, color=colors[i], alpha=0.1)

        if cutOff[-1] >= longestRunPos:
            longestRunPos = cutOff[-1]

        i += 1

    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.set_title(title)
    plt.ylim(limits)
    plt.xlim(right=longestRunPos+5)
    plt.legend()
    if saveName:
        plt.savefig(f'{savePath}{saveName}.png')
    plt.show()
    return


# Total best performance
def visualizeTotalPerformance(performances: dict, listOrder: list, title="Framework performance", saveName="", loc='best', inPercent=True, plotKaggle=False, limits=None, logScale=False):
    def getPecentIncresement(old, new, percent=True):
        if percent:
            return (new-old)/old*100
        else:
            return new

    def getPerformance(performanceDict, modelName, percentage):
        performancesForModel = [performanceDict[frameworkName].get(modelName) for frameworkName in frameworkOrder]
        nonePerformance = performanceDict["None"][modelName]
        if percentage:
            return [[getPecentIncresement(nonePerformance, perf) for perf in performanceModel] for performanceModel in performancesForModel]
        else:
            return performancesForModel

    # don't plot frameworks with temp in name
    for key in performances.copy().keys():
        if "temp" in key:
            del performances[key]

    # no naming conventions during runs:
    performances["Auto-sklearn"]["LinearRegression"] = performances["Auto-sklearn"].pop("LinearReg")
    performances["TPOT"]["DecisionTree"] = performances["TPOT"].pop("DecisionTreeRegressor")
    performances["TPOT"]["RandomForest"] = performances["TPOT"].pop("RandomForestRegressor")

    frameworkOrder = [ "Auto-sklearn", "Autofeat","TPOT"]
    labels = ["auto-sklearn","autofeat","TPOT"]

    fs = 12  # fontsize

    # TODO test figsizes
    plt.style.use('seaborn-whitegrid')
    fig, axes = plt.subplots(nrows=4, ncols=1, sharex=True, dpi=200)

    if logScale:
        for ax in axes:
            ax.set_xscale('log')

    i = 0
    for modelName in listOrder:
        bp_dict = axes[i].boxplot(getPerformance(performances, modelName, inPercent), labels=labels, showmeans=False, meanline=True, vert=False, showfliers=False)
        if not inPercent:
            percentIncreasement = getPerformance(performances, modelName, True)
            j = 0
            for line in bp_dict['medians']:
                x, y = line.get_xydata()[1]  # top of median line
                # overlay percent improvement
                axes[i].annotate(int(np.median(percentIncreasement[j])), (x,y+0.02), horizontalalignment='center')
                j += 1

        if plotKaggle:
            axes[i].axvline(getPecentIncresement(performances["None"][modelName], performances["None"][modelName], inPercent), c="r", ls="--", label="No Engineering")
            if not "Synthetic" in title:
                axes[i].axvline(getPecentIncresement(performances["None"][modelName], performances["Kaggle"][modelName], inPercent), c="g", ls="--", label="Kaggle Solution")
        axes[i].set_title(modelName, fontsize=fs)
        i += 1

    #fig.suptitle(title)
    #fig.text(0.04, 0.5, 'Framework', va='center', rotation='vertical')
    if inPercent:
        fig.text(0.6, 0.0, 'RMSE improvement in percent.', ha='center')
    else:
        fig.text(0.6, 0.0, 'RMSE', ha='center')

    fig.tight_layout()
    plt.xlim(limits)
    axes[0].legend(loc=loc)

    if saveName:
        saveNameAddition=""
        if inPercent:
            saveNameAddition = saveNameAddition + "_percent"
        plt.savefig(f'{savePath}{saveName}{saveNameAddition}.png')
    plt.show()
    return

# Autofeat plotting
def visualizeAutofeatPerformance(dataName, default=None, saveName="", vmax=None):
    def heatmap(data, row_labels, col_labels, ax=None,
                cbar_kw={}, cbarlabel="", **kwargs):

        if not ax:
            ax = plt.gca()

        # Plot the heatmap
        im = ax.imshow(data, **kwargs)

        # Create colorbar
        cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
        cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

        # We want to show all ticks...
        ax.set_xticks(np.arange(data.shape[1]))
        ax.set_yticks(np.arange(data.shape[0]))
        # ... and label them with the respective list entries.
        ax.set_xticklabels(col_labels)
        ax.set_yticklabels(row_labels)

        # Let the horizontal axes labeling appear on top.
        ax.tick_params(bottom=True, labelbottom=True)

        ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
        ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
        ax.tick_params(which="minor", bottom=False, left=False)

        ax.set_xlabel("Selection steps")
        ax.set_ylabel("Generation steps")

        return im, cbar

    def annotate_heatmap(im, data=None,
                         textcolors=["black", "white"],
                         threshold=None, **textkw):

        if not isinstance(data, (list, np.ndarray)):
            data = im.get_array()

        # Normalize the threshold to the images color range.
        if threshold is not None:
            threshold = im.norm(threshold)
        else:
            threshold = im.norm(data.max())/2.

        # Set default alignment to center, but allow it to be
        # overwritten by textkw.
        kw = dict(horizontalalignment="center",
                  verticalalignment="center")
        kw.update(textkw)

        # Loop over the data and create a `Text` for each "pixel".
        # Change the text's color depending on the data.
        texts = []
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                #kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
                text = im.axes.text(j, i, int(data[i, j]), **kw)
                texts.append(text)

        return texts

    fig, ((ax, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 6), dpi=200)
    plt.style.use('default')
    axes = [ax,ax2,ax3,ax4]

    i = 0
    featEngSteps = [1,2,3]

    performances_autofeat, featureCount = getPerformances_autofeat(dataName, inPercent=default["None"])

    for name in modelOrder:
        axes[i].set_title(name)
        im, cbar = heatmap(np.array(performances_autofeat[name]), featEngSteps, [1,2,3,4,5], ax=axes[i],
                           cmap="GnBu", cbarlabel="Median percent RMSE improvement", vmin=0, vmax=vmax)
        texts = annotate_heatmap(im, data=np.array(featureCount[name]))
        i += 1

    fig.tight_layout()
    #plt.suptitle("Title centered above all subplots", fontsize=16)
    if saveName:
        plt.savefig(f'{savePath}{saveName}.png')
    plt.show()
    return


# for visualisation of total performances
modelOrder = ["RandomForest", "DecisionTree", "LinearRegression", "LassoLarsCV"]

# Again,  10.000 samples of datasets, tested on no FE and kaggle FE
# For the Kaggle performances, reimplement/copy the FE of the cited Kaggle solutions

syntheticPerformance = {"None": {"DecisionTree":95284,
                                "RandomForest":64342,
                                "LinearRegression":43587280,
                                "LassoLarsCV":43580720}}

rossmannPerformance = {"None": {"DecisionTree":967,
                                "RandomForest":714,
                                "LinearRegression":1394,
                                "LassoLarsCV":1394},
                       "Kaggle": {"DecisionTree": 163,
                                "RandomForest": 129,
                                "LinearRegression": 905,
                                "LassoLarsCV": 905}}

taxitripPerformance = {"None": {"DecisionTree":350,
                                "RandomForest":251,
                                "LinearRegression":440,
                                "LassoLarsCV":445},
                       "Kaggle": {"DecisionTree": 314,
                                "RandomForest": 227,
                                "LinearRegression": 392,
                                "LassoLarsCV": 380}}

# Get graphs of optimization history
#%% tpot
visualiseBestSoFar(getPerformanceHistory_TPOT("synthetic_0.7", byGeneration=True), x_label="Generation", y_label="RMSE", title="TPOT performance for Synthetic", saveName="tpot_synthetic", limits=(10**1, 10**7), logScale=True)
visualiseBestSoFar(getPerformanceHistory_TPOT("rossmann_0.7", byGeneration=True), x_label="Generation", y_label="RMSE", title="TPOT performance for Rossmann", saveName="tpot_rossmann", limits=(690, 1240))
visualiseBestSoFar(getPerformanceHistory_TPOT("taxitrip_0.7", byGeneration=True), x_label="Generation", y_label="RMSE", title="TPOT performance for TaxiTrip", saveName="tpot_taxitrip", limits=(240, 460))

#%%  autosklearn
visualiseBestSoFar(getPerformanceHistory_autosklearn("synthetic_0.7"), y_label="RMSE", x_label="Evaluation Number", title="Autos-sklearn performance for Synthetic", saveName="autosklearn_synthetic", logScale=True)
visualiseBestSoFar(getPerformanceHistory_autosklearn("rossmann_0.7"), y_label="RMSE", x_label="Evaluation number", title="Auto-sklearn performance for Rossmann", saveName="autosklearn_rossmann")
visualiseBestSoFar(getPerformanceHistory_autosklearn("taxitrip_0.7"), y_label="RMSE", x_label="Evaluation Number", title="Autos-sklearn performance for TaxiTrip", saveName="autosklearn_taxitrip")

#%% AutofeatPlots
visualizeAutofeatPerformance("synthetic_0.7", default=syntheticPerformance, saveName="autofeat_synthetic", vmax=100)
visualizeAutofeatPerformance("rossmann_0.7", default=rossmannPerformance, saveName="autofeat_rossmann", vmax=30)
visualizeAutofeatPerformance("taxitrip_0.7", default=taxitripPerformance, saveName="autofeat_taxitrip", vmax=30)

#%% Total performance over all
visualizeTotalPerformance(getTotalPerformances("synthetic_0.7", default=syntheticPerformance), modelOrder, title="Framework performance for Synthetic", saveName="total_Synthetic", inPercent=False, plotKaggle=True, logScale=True, loc='upper right')
visualizeTotalPerformance(getTotalPerformances("rossmann_0.7", default=rossmannPerformance), modelOrder, title="Framework performance for Rossmann", saveName="total_rossmann", inPercent=False, plotKaggle=True, loc='upper right')
visualizeTotalPerformance(getTotalPerformances("taxitrip_0.7", default=taxitripPerformance), modelOrder, title="Framework performance for TaxiTrip", saveName="total_taxitrip", inPercent=False, plotKaggle=True, loc='upper right')


#%%  plot target data distribution
_,_,targetValues_taxitrip_def,_ = DataHandler.getData_taxitrip(trainSize=0.9999)
_,_,targetValues_taxitrip,_ = DataHandler.getData_taxitrip(trainSize=0.9999, preprocessed=True)

fig, axs = plt.subplots(2,1, dpi=200)

axs[0].boxplot(targetValues_taxitrip_def, vert=False)
axs[0].set_title("Target distribution")
axs[0].set_yticklabels([""])
axs[0].set_xlabel("Seconds")

axs[1].boxplot(targetValues_taxitrip, vert=False)
axs[1].set_title(" Target distribution after removing outliers")
axs[1].set_yticklabels([""])
axs[1].set_xlabel("Seconds")

plt.tight_layout(rect=(0,0,1,0.6))
plt.show()

#%% Xor visualization
xor=pd.DataFrame(data=[[0,0,0,1],[1,1,0,1],[0,1,1,-1],[1,0,1,-1]], columns=["x","y","target","xnew"])

plt.scatter(xor.x.where(xor.target==0), xor.y.where(xor.target==0), alpha=0.8, c="blue", s=100, edgecolors='none', label="0")
plt.scatter(xor.x.where(xor.target==1), xor.y.where(xor.target==1), alpha=0.8, c="red",s=100, edgecolors='none', label="1")

# plot necessary linear seperators
data = np.asarray([-2, -1, 0, 1, 2])
plt.plot(data, data-.5, c="lightskyblue", linestyle="--")
plt.plot(data, data+.5, c="lightskyblue", linestyle="--")

plt.xticks(np.arange(-1, 3), ["","0","1",""])
plt.yticks(np.arange(-1, 3), ["","0","1",""])
plt.ylabel("x2")
plt.xlabel("x1")
plt.xlim(right=1.75,left=-.75)
plt.ylim(top=1.75,bottom=-.75)
#plt.clim(0, 1)
plt.title("XOR")
plt.legend(loc=2)
plt.show()

#after transformation#
plt.scatter(xor.xnew.where(xor.target==0), [0,0,0,0], alpha=0.8, c="blue", s=100, edgecolors='none', label="0")
plt.scatter(xor.xnew.where(xor.target==1), [0,0,0,0], alpha=0.8, c="red",s=100, edgecolors='none', label="1")

# plot necessary lin seperators
data = np.asarray([-2, -1, 0, 1, 2])
plt.plot([0,0,0,0,0], data, c="lightskyblue", linestyle="--")

plt.yticks([0],[""])
plt.xticks(np.arange(-2, 3), ["","-1","0","1",""])
plt.ylim(top=.5,bottom=-.5)
plt.ylabel("")
plt.xlabel("xnew")
plt.title("Feature transformed XOR")
plt.legend(loc=2)
plt.show()


#%% visualise TPOT individuals and operator information
operatorsHistory, operatorCount_s, nindividualsHistory, nindividualsCount_s, bestOperatorCount_s = getPerformanceHistory_TPOT("synthetic_0.7", byGeneration=True, returnCount=True, verbosity=True)
operatorsHistory, operatorCount_r, nindividualsHistory, nindividualsCount_r, bestOperatorCount_r = getPerformanceHistory_TPOT("rossmann_0.7", byGeneration=True, returnCount=True, verbosity=True)
operatorsHistory, operatorCount_t, nindividualsHistory, nindividualsCount_t, bestOperatorCount_t = getPerformanceHistory_TPOT("taxitrip_0.7", byGeneration=True, returnCount=True, verbosity=True)
operatorCount = [operatorCount_s, operatorCount_r, operatorCount_t]
nindividualsCount = [nindividualsCount_s, nindividualsCount_r, nindividualsCount_t]
bestOperatorCount = [bestOperatorCount_s, bestOperatorCount_r, bestOperatorCount_t]
labelNames = ["Synthetic", "Rossmann", "Taxitrip"]
#%%
fig, ax = plt.subplots(dpi=200)
ax.hist(nindividualsCount, bins=range(1,50), density=True, histtype='bar', stacked=True, color=colors[1:4], label=labelNames)
ax.set_title("New individuals over generations")
fig.tight_layout()
plt.ylabel("Probability density")
plt.xlabel("Number of new individuals")
plt.legend()
plt.savefig(f'{savePath}tpot_individuals_generations.png')
plt.show()

fig, ax = plt.subplots(dpi=200)
ax.hist(operatorCount, bins=range(1,13), density=True, histtype='bar', stacked=True, color=colors[1:4], label=labelNames)
ax.set_title('Number of operators in all pipelines')
plt.ylim(0, 0.59)
plt.ylabel("Probability density")
plt.xlabel("Number of operators")
fig.tight_layout()
plt.legend()
plt.savefig(f'{savePath}tpot_operators_pipeline.png')
plt.show()

fig, ax = plt.subplots(dpi=200)
ax.hist(bestOperatorCount, bins=range(1,13), density=True, histtype='bar', stacked=True, color=colors[1:4], label=labelNames)
ax.set_title('Number of operators in best pipelines')
plt.ylim(0, 0.59)
plt.ylabel("Probability density")
plt.xlabel("Number of operators")
fig.tight_layout()
plt.legend()
plt.savefig(f'{savePath}tpot_operators_bestpipeline.png')
plt.show()

print(np.median([item for sublist in operatorCount for item in sublist]))
print(np.mean([item for sublist in operatorCount for item in sublist]))
print(np.median([item for sublist in bestOperatorCount for item in sublist]))
print(np.mean([item for sublist in bestOperatorCount for item in sublist]))


#%% Plot heatmap for operator count in best pipelines
def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw, ticks=range(0,21,5))
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(bottom=True, labelbottom=True)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    return im, cbar

xaxisNames = ["RF Syn.","DT Syn.", "LR Syn.", "LL Syn.",
              "RF Ross.","DT Ross.", "LR Ross.", "LL Ross.",
              "RF Taxi.","DT Taxi.", "LR Taxi.", "LL Taxi."]

fig, axes = plt.subplots(dpi=200)
plt.style.use('default')
axes.set_title("Operator occurrence")
operatorMatrix, pipelineOperators = operatorsUsed()
im, cbar = heatmap(np.array(operatorMatrix), pipelineOperators, xaxisNames, ax=axes,
                   cmap="GnBu", cbarlabel="Number of occurrence", vmax=20)
fig.tight_layout()
plt.savefig(f'{savePath}tpot_operator_occurrence.png')
plt.show()

fig, axes = plt.subplots(dpi=200)
plt.style.use('default')
axes.set_title("Operator occurrence")
operatorMatrix, pipelineOperators = operatorsUsed(False)
im, cbar = heatmap(np.array(operatorMatrix), pipelineOperators, xaxisNames, ax=axes,
                   cmap="GnBu", cbarlabel="Number of occurrence", vmax=20)
fig.tight_layout()
plt.savefig(f'{savePath}autosklearn_operator_occurrence.png')
plt.show()



#%%
data1 = []
data1.append(getMedianPerformance(dataName="synthetic_0.7", framework="TPOT"))
data1.append(getMedianPerformance(dataName="synthetic_0.7", framework="autosklearn"))
data1.append(getMedianPerformance(dataName="synthetic_0.7", framework="autofeat"))
# Kaggle
# None
data1.append([64342,95284,43587280,43580720])
data1 = np.array(data1).transpose().tolist()


data2 = []
data2.append(getMedianPerformance(dataName="rossmann_0.7", framework="TPOT"))
data2.append(getMedianPerformance(dataName="rossmann_0.7", framework="autosklearn"))
data2.append(getMedianPerformance(dataName="rossmann_0.7", framework="autofeat"))
# Kaggle
data2.append([129,163,905,905])
# None
data2.append([714,967,1394,1394])
data2 = np.array(data2).transpose().tolist()

data3 = []
data3.append(getMedianPerformance(dataName="taxitrip_0.7", framework="TPOT"))
data3.append(getMedianPerformance(dataName="taxitrip_0.7", framework="autosklearn"))
data3.append(getMedianPerformance(dataName="taxitrip_0.7", framework="autofeat"))
# Kaggle
data3.append([227,314,392,380])
# None
data3.append([251,350,440,445])
data3 = np.array(data3).transpose().tolist()


#%%
fig, axes = plt.subplots(3,1, dpi=300)
ypositions = [0.9, 0.7, 0.5, 0.3]
mark1 = ["s", "p", "h", "o"]
ypositions2 = [0.9, 0.75, 0.6, 0.45, 0.3]
mark2 = ["s", "p", "h", "*", "o"]

# per data
i = 0
for dataSet, ax, title in zip([data1, data2, data3], axes, ["Synthetic", "Rossmann", "TaxiTrip"]):
    i += 1
    if i==1:
        posF=ypositions
        mark = mark1
    else:
        posF=ypositions2
        mark = mark2

    # per model
    for elem, color in zip(dataSet, colors):
        # per framework
        for stem, offset in zip(elem, posF):
            markerline, stemlines, baseline = ax.stem([stem], [offset], ':', markerfmt="o")
            plt.setp(markerline, markersize = 6, color=color, markerfacecolor=color)
            plt.setp(stemlines, color = color, linewidth = 1)
    if i==1:
        ax.set_xscale("log")
        ax.set_xlim(left=10)
        ax.set_yticklabels(["TPOT","auto-sklearn","autofeat","None"])
    else:
        ax.set_yticklabels(["TPOT","auto-sklearn","autofeat","Kaggle","None"])
    ax.set_yticks(posF)
    ax.set_ylim(top=1, bottom=0.1)
    ax.set_title(title)

legend_elements = [Line2D([0], [0], marker='o', color='w', label='RandomForest', markerfacecolor=colors[0], markersize=8),
                   Line2D([0], [0], marker='o', color='w', label='DecisionTree', markerfacecolor=colors[1], markersize=8),
                   Line2D([0], [0], marker='o', color='w', label='LinearRegression', markerfacecolor=colors[2], markersize=8),
                   Line2D([0], [0], marker='o', color='w', label='LassoLarsCV', markerfacecolor=colors[3], markersize=8)]


axes[2].legend(handles=legend_elements, loc="lower center", ncol=4, bbox_to_anchor=(0.45, -0.7), columnspacing=0.1,
               fontsize="medium", borderpad=0, edgecolor="white")
plt.savefig(f'{savePath}total_overall.png')
plt.show()

