import time
import Data
from Model import Regression
import Result

#def TestClassification():
#    trainSet    = Classification.LoadSet(["train"])
#    easyTestSet = Classification.LoadSet(["test_easy"])
#    hardTestSet = Classification.LoadSet(["test_hard"])
#
#    print("Creation model for regression...")
#    
#    model = ModelClassification() 
#    model.Create()
#    
#    print("Training model for regression...")
#
#    #history = model.Train(trainSet, 'sgd',  1000) #1000
#    history = model.Train(trainSet, 'adam', 1000, 0) #1000
#    #PlotResult(history, "accuracy")
#    
#    print("Evaluating model for regression...")
#    
#    scoreTrain = model.Evaluate(trainSet)
#    scoreEasy  = model.Evaluate(easyTestSet)
#    scoreHard  = model.Evaluate(hardTestSet)   
#    PrintAssertClassification(scoreTrain[1], "Train")
#    PrintAssertClassification(scoreEasy[1],  "Easy")
#    PrintAssertClassification(scoreHard[1],  "Hard")
#    return (model, trainSet, easyTestSet, hardTestSet)

def TestRegression():
    trainSet    = Data.Regression.Regression.LoadSet(["train", "train_auto-generated"])#2 , "train_auto-generated"
    easyTestSet = Data.Regression.Regression.LoadSet(["test_easy"])#2
    hardTestSet = Data.Regression.Regression.LoadSet(["test_hard"])#2

    print("Creation model for regression...")
    
    model = Regression.RegressionModel()#2
    model.Create()
    
    print("Training model for regression...")
    
    while True:
        epoch = int(input("Number of epoch: "))
        if epoch == 0:
            break
        #history = model.Train(trainSet, 'sgd', 100, 2) #3000 #200
        #PlotResult(history, "mae")
        history = model.Train(trainSet, 'adam', epoch, 2) #3000 #200
        PlotResult(history, 'accuracy')
        #history = model.Train(trainSet, 'sgd', epoch, 2) #3000 #200
        #PlotResult(history, 'accuracy')    

    print("Evaluating model for regression...")
    
    scoreTrain = model.Evaluate(trainSet)
    scoreEasy = model.Evaluate(easyTestSet)
    scoreHard = model.Evaluate(hardTestSet)
    
    PrintAssertRegression(scoreTrain[1], "Train")#2
    PrintAssertRegression(scoreEasy[1], "Easy")#2
    PrintAssertRegression(scoreHard[1], "Hard")#2
    return model


if __name__ == "__main__":

    print("Start")
    start = time.time()
    #c = TestClassification()
    model = TestRegression()
    print("========================================================================")
    print("========================================================================")
    #GlobalEvaluation(model)#2
    print("Run in", round(time.time() - start), "secondes")
    print("End")
