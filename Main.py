from ModelRegression import *
from ModelClassification import *
from DataClassification import *
from DataRegression import *
from Plot import *
from Assert import *

def regression():
    trainSet = Regression.LoadSet("train")
    print(trainSet[0].shape)
    easyTestSet = Regression.LoadSet("test_easy")
    hardTestSet = Regression.LoadSet("test_hard")

    print("Creation model for regression...")
    
    model = ModelRegression() 
    model.Create()
    
    print("Training model for regression...")
    
    history = model.Train(trainSet, 3)
    PlotResult(history, "mae")
    print("Evaluating model for regression...")
    
    scoreEasy = model.Evaluate(easyTestSet)
    scoreHard = model.Evaluate(hardTestSet)
    
    PrintAssertRegression(scoreEasy[1], "Easy")
    PrintAssertRegression(scoreHard[1], "Hard")

def classification():
    trainSet = Classification.LoadSet("train")
    print(trainSet[0].shape)
    easyTestSet = Classification.LoadSet("test_easy")
    hardTestSet = Classification.LoadSet("test_hard")

    print("Creation model for regression...")
    
    model = ModelClassification() 
    model.Create()
    
    print("Training model for regression...")
    
    history = model.Train(trainSet, 1000)
    PlotResult(history, "accuracy")
    
    print("Evaluating model for regression...")
    
    scoreTrain = model.Evaluate(trainSet)
    scoreEasy  = model.Evaluate(easyTestSet)
    scoreHard  = model.Evaluate(hardTestSet)   
    PrintAssertClassification(scoreTrain[1], "train")
    PrintAssertClassification(scoreEasy[1],  "Easy")
    PrintAssertClassification(scoreHard[1],  "Hard")

if __name__ == "__main__":

    print("Start")
    classification()
    #print("========================================================================")
    #print("========================================================================")
    #regression()
    print("End")
