import time
import tools as tl
import data
from model import regression
import result

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
    train_set = data.Regression.Regression.LoadSet(["train", "train_auto-generated"])
    easy_test_set = data.Regression.Regression.LoadSet(["test_easy"])
    hard_test_set = data.Regression.Regression.LoadSet(["test_hard"])

    print("Creation model for regression...")
    
    model = regression.Model()#2
    model.Create()
    
    print("Training model for regression...")
    
    while True:
        epoch = tl.Int(input("Number of epoch: "))
        if epoch == 0:
            break
        #history = model.Train(trainSet, 'sgd', 100, 2) #3000 #200
        #PlotResult(history, "mae")
        history = model.Train(train_set, 'adam', epoch, 2) #3000 #200
        result.plot.display(history, 'accuracy')
        #history = model.Train(trainSet, 'sgd', epoch, 2) #3000 #200
        #PlotResult(history, 'accuracy')    

    print("Evaluating model for regression...")
    
    score_train = model.Evaluate(train_set)
    score_easy = model.Evaluate(easy_test_set)
    score_hard = model.Evaluate(hard_test_set)
    
    result.print.regression(score_train[1], "Train")
    result.print.regression(score_easy[1], "Easy")
    result.print.regression(score_hard[1], "Hard")
    return model

def main():
    print("Start")
    start = time.time()
    #c = TestClassification()
    model = TestRegression()
    print("========================================================================")
    print("========================================================================")
    GlobalEvaluation(model)
    print("Run in", round(time.time() - start), "secondes")
    print("End")
    
if __name__ == "__main__":
    main()