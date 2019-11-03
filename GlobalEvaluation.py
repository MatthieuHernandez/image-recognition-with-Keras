import numpy as np
import operator
from Data import *
from Assert import *
from DataRegression import *

Cooldowns = [ v for v in SpellCooldowns.values() ]

def GlobalEvaluation(model):

    trainSet    = Regression.LoadSet(["train"])
    easyTestSet = Regression.LoadSet(["test_easy"])
    hardTestSet = Regression.LoadSet(["test_hard"])

    labelsTrain  = LoadLabels("train")
    labelsEasy   = LoadLabels("test_easy")
    labelsHard   = LoadLabels("test_hard")

    accuracyTrain = Evaluation(model, trainSet,     labelsTrain)
    accuracyEasy  = Evaluation(model, easyTestSet,  labelsEasy)
    accuracyHard  = Evaluation(model, hardTestSet,  labelsHard)

    PrintAssertClassification(accuracyTrain, "train")
    PrintAssertClassification(accuracyEasy,  "Easy")
    PrintAssertClassification(accuracyHard,  "Hard")


def Evaluation(model, setR, labels):

    well = 0
    bad = 0
    for i in range(0, len(setR[0])):
        inputs = setR[0][i].reshape(1, 20, 20, 1)
        result = model.Predict(inputs)[0][0] * 100
        #print(result, "=", labels[i])
        if abs(result - labels[i]) < 2 :
            well = well + 1
        else :
            bad = bad + 1
    accuracy = well/(well+bad)
    return accuracy
        

def LoadLabels(folder):
    labels = LoadJson(folder)
    sorted_labels = sorted(labels.items(), key=operator.itemgetter(0))
    values = [x[1] for x in sorted_labels]
    return np.asarray(values)
