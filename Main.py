from Model import *
from Data import *


if __name__ == "__main__":

    print("Start")

    trainSet = LoadSet("train")
    easyTestSet = LoadSet("test_easy")
    hardTestSet = LoadSet("test_hard")

    model = Model()
    print("Creation...")
    model.Create()
    print("Compling...")
    model.Compile()
    print("Training...")
    model.Train(trainSet)
    
    print("End")
