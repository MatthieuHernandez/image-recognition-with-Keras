from Model import *
from Data import *
from Plot import *

if __name__ == "__main__":

    print("Start")

    trainSet = LoadSet("train")
    easyTestSet = LoadSet("test_easy")
    hardTestSet = LoadSet("test_hard")

    model = Model()
    print("Creation...")
    model.Create()
    print("Training...")
    history = model.Train(trainSet)
    PlotResultLoss(history)
    print("Evaluating...")
    score = model.Evaluate(easyTestSet)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    print("End")
