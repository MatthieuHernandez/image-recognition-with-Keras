import numpy as np
import operator
from Data import *
from Assert import *

Cooldowns = [ v for v in SpellCooldowns.values() ]

def GlobalEvaluation(classification, Regression):  
    modelC    = classification[0]
    setTrainC = classification[1]
    setEasyC  = classification[2]
    setHardC  = classification[3]

    modelR    = Regression[0]
    setTrainR = Regression[1]
    setEasyR  = Regression[2]
    setHardR  = Regression[3]

    labelsTrain = LoadLabels("train")
    labelsEasy  = LoadLabels("test_easy")
    labelsHard  = LoadLabels("test_hard")

    accuracyTrain = Evaluation(modelC, modelR, setTrainC, setTrainR, labelsTrain)
    accuracyEasy  = Evaluation(modelC, modelR, setEasyC,  setEasyR,  labelsEasy)
    accuracyHard  = Evaluation(modelC, modelR, setHardC,  setHardR,  labelsHard)

    PrintAssertClassification(accuracyTrain, "train")
    PrintAssertClassification(accuracyEasy,  "Easy")
    PrintAssertClassification(accuracyHard,  "Hard")


def Evaluation(modelC, modelR, setC, setR, labels):

    well = 0
    bad = 0
    for i in range(0, len(setC[0])):

        inputs = setC[0][i].reshape(1, 1200)
        outputs = np.array(modelC.Predict(inputs))[0]
        predictedClass = list(outputs).index(max(outputs))
        duration = Cooldowns[predictedClass]

        inputs = setR[0][i].reshape(1, 20, 20, 3)
        percentageDuration = modelR.Predict(inputs)[0][0]
        result = round(duration * percentageDuration)
        #print(duration, " * ", percentageDuration, " = ", result, " = ", labels[i])
        if abs(result - labels[i]) <= 1 :
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
