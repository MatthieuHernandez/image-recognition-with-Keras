from ModelRegression import *
from ModelClassification import *
from DataClassification import *
from DataRegression import *
from Plot import *
from Assert import *
from GlobalEvaluation import *
import time

def TestClassification():
    trainSet    = Classification.LoadSet(["train"])
    easyTestSet = Classification.LoadSet(["test_easy"])
    hardTestSet = Classification.LoadSet(["test_hard"])

    print("Creation model for regression...")
    
    model = ModelClassification() 
    model.Create()
    
    print("Training model for regression...")

    #history = model.Train(trainSet, 'sgd',  1000) #1000
    history = model.Train(trainSet, 'adam', 1000, 0) #1000
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
    trainSet    = Regression.LoadSet(["train", "train_fake"])
    easyTestSet = Regression.LoadSet(["test_easy"])
    hardTestSet = Regression.LoadSet(["test_hard"])

    print("Creation model for regression...")
    
    model = ModelRegression() 
    model.Create()
    
    print("Training model for regression...")
    
    history = model.Train(trainSet, 'sgd', 35, 0) #3000 #200
    #PlotResult(history, "mae")
    history = model.Train(trainSet, 'adam', 300, 2) #3000 #200
    #PlotResult(history, "mae")
    history = model.Train(trainSet, 'sgd', 200, 2) #3000 #200
    #PlotResult(history, "mae")
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
    #c = TestClassification()
    print("========================================================================")
    print("========================================================================")
    r = TestRegression()
    print("========================================================================")
    print("========================================================================")
    #GlobalEvaluation(c, r)
    print("Run in ",round(time.time() - start),"secondes")
    print("End")
