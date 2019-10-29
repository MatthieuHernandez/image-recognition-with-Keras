from ModelRegression import *
from ModelClassification import *
from DataClassification import *
from DataRegression import *
from Plot import *
from Assert import *
from GlobalEvaluation import *
import time

def TestClassification():
    trainSet    = Classification.LoadSet("train")
    easyTestSet = Classification.LoadSet("test_easy")
    hardTestSet = Classification.LoadSet("test_hard")

    print("Creation model for regression...")
    
    model = ModelClassification() 
    model.Create()
    
    print("Training model for regression...")
    
    history = model.Train(trainSet, 1000) #1000
    #PlotResult(history, "accuracy")
    
    print("Evaluating model for regression...")
    
    scoreTrain = model.Evaluate(trainSet)
    scoreEasy  = model.Evaluate(easyTestSet)
    scoreHard  = model.Evaluate(hardTestSet)   
    PrintAssertClassification(scoreTrain[1], "Train")
    PrintAssertClassification(scoreEasy[1],  "Easy")
    PrintAssertClassification(scoreHard[1],  "Hard")
    return (model, trainSet, easyTestSet, hardTestSet)


def TestRegression():
    trainSet    = Regression.LoadSet("train")
    easyTestSet = Regression.LoadSet("test_easy")
    hardTestSet = Regression.LoadSet("test_hard")

    print("Creation model for regression...")
    
    model = ModelRegression() 
    model.Create()
    
    print("Training model for regression...")
    
    history = model.Train(trainSet, 900) # 3000
    PlotResult(history, "mae")
    print("Evaluating model for regression...")

    scoreTrain = model.Evaluate(trainSet)
    scoreEasy = model.Evaluate(easyTestSet)
    scoreHard = model.Evaluate(hardTestSet)
    PrintAssertRegression(scoreTrain[1], "Train")
    PrintAssertRegression(scoreEasy[1], "Easy")
    PrintAssertRegression(scoreHard[1], "Hard")
    return (model, trainSet, easyTestSet, hardTestSet)


if __name__ == "__main__":

    print("Start")
    start = time.time()
    c = TestClassification()
    print("========================================================================")
    print("========================================================================")
    r = TestRegression()
    print("========================================================================(")
    print("========================================================================")
    GlobalEvaluation(c, r)
    print("End")
