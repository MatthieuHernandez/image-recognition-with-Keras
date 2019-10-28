from Model import *
from Data import *
from Plot import *
from Assert import *



if __name__ == "__main__":

    print("Start")

    trainSet = LoadSet("train")
    print(trainSet[0].shape)
    easyTestSet = LoadSet("test_easy")
    hardTestSet = LoadSet("test_hard")

    print("Creation...")
    
    model = Model() 
    model.Create()
    
    print("Training...")
    
    history = model.Train(trainSet)
    PlotResult(history, "mae")
    print("Evaluating...")
    
    scoreEasy = model.Evaluate(easyTestSet)
    scoreHard = model.Evaluate(hardTestSet)
    
    PrintAssert(scoreEasy[1], "Easy")
    PrintAssert(scoreHard[1], "Hard")
    
    print("End")
