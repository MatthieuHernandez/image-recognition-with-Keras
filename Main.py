import time

import result
from data.regression import Regression
from model import regression
from tools import *

auto_stop = True


def test_regression():
    train_set = Regression.load_set(
        ["train", "train_auto-generated"])  # "train",
    easy_test_set = Regression.load_set(["test_easy"])
    hard_test_set = Regression.load_set(["test_hard"])

    print("Creation model for regression...")

    model = regression.CustomModel('complexe')

    print("Training model for regression...")
    old_mae = 1
    new_mae = 1
    if auto_stop:
        while new_mae <= old_mae and new_mae > 0.02:
            history = model.train(train_set, 'adam', 20, 0)
            score_hard = model.evaluate(hard_test_set)
            old_mae = new_mae
            new_mae = score_hard[1]
            print("mae =", round(score_hard[1], 4))
    else:
        while True:
            epoch = to_int(input("Number of epoch: "))  # 110
            if epoch == 0:
                break
            history = model.train(train_set, 'adam', epoch, 2)
            result.plot.display(history, 'mae')
            # history = model.Train(trainSet, 'sgd', epoch, 2)
            #PlotResult(history, 'accuracy')

    print("Evaluating model for regression...")

    score_train = model.evaluate(train_set)
    score_easy = model.evaluate(easy_test_set)
    score_hard = model.evaluate(hard_test_set)

    result.display.regression(score_train[1], "Train")
    result.display.regression(score_easy[1], "Easy")
    result.display.regression(score_hard[1], "Hard")
    return model


def main():
    print("Start")
    start = time.time()
    #c = TestClassification()
    model = test_regression()
    print("========================================================================")
    print("========================================================================")
    result.global_evaluation.global_evaluation(model)
    print("Run in", round(time.time() - start), "secondes")
    print("End")


if __name__ == "__main__":
    main()
