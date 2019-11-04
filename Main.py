from ModelRegression import *
from ModelRegression2 import *
from DataRegression import *
from DataRegression2 import *
from ModelClassification import *
from DataClassification import *
from Plot import *
from Assert import *
from GlobalEvaluation import *
from GlobalEvaluation2 import *
import time

'''def TestClassification():
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
    return (model, trainSet, easyTestSet, hardTestSet)'''


def TestRegression():
    trainSet    = Regression2.LoadSet(["train", "train_auto-generated"])
    easyTestSet = Regression2.LoadSet(["test_easy"])
    hardTestSet = Regression2.LoadSet(["test_hard"])

    print("Creation model for regression...")
    
    model = ModelRegression2()
    model.Create()
    
    print("Training model for regression...")
    
    #history = model.Train(trainSet, 'sgd', 35, 2) #3000 #200
    #PlotResult(history, "mae")
    history = model.Train(trainSet, 'adam', 200, 2) #3000 #200
    #PlotResult(history, 'accuracy')
    history = model.Train(trainSet, 'sgd', 40, 2) #3000 #200
    #PlotResult(history, 'accuracy')
    print("Evaluating model for regression...")
    
    scoreTrain = model.Evaluate(trainSet)
    scoreEasy = model.Evaluate(easyTestSet)
    scoreHard = model.Evaluate(hardTestSet)
    
    PrintAssertRegression2(scoreTrain[1], "Train")
    PrintAssertRegression2(scoreEasy[1], "Easy")
    PrintAssertRegression2(scoreHard[1], "Hard")
    return model


if __name__ == "__main__":

    print("Start")
    start = time.time()
    #c = TestClassification()
    model = TestRegression()
    print("========================================================================")
    print("========================================================================")
    GlobalEvaluation2(model)
    print("Run in",round(time.time() - start),"secondes")
    print("End")
