import operator

import numpy as np

import result
from data.regression import *


def global_evaluation(model):

    train_set = Regression.load_set(["train"])
    easy_test_set = Regression.load_set(["test_easy"])
    hard_test_set = Regression.load_set(["test_hard"])

    labels_train = load_labels("train")
    labels_easy = load_labels("test_easy")
    labels_hard = load_labels("test_hard")

    accuracy_train = evaluation(model, train_set, labels_train)
    accuracy_easy = evaluation(model, easy_test_set, labels_easy)
    accuracy_hard = evaluation(model, hard_test_set, labels_hard, True)

    result.display.regression(accuracy_train, "train")
    result.display.regression(accuracy_easy, "Easy")
    result.display.regression(accuracy_hard, "Hard")


def evaluation(model, dataset, labels, print_errors=False):

    well = 0
    bad = 0
    for i in range(0, len(dataset[0])):
        inputs = dataset[0][i].reshape(1, 20, 20, 1)
        res = model.predict(inputs)[0][0] * 100
        if abs(res - labels[i]) < 5:  # 2
            well = well + 1
        else:
            if print_errors:
                print("{0:02f}".format(res), "=",
                      labels[i], "    index=", i)
            bad = bad + 1
    accuracy = well/(well+bad)
    return accuracy


def load_labels(folder):
    labels = load_json(folder)
    sorted_labels = sorted(labels.items(), key=operator.itemgetter(0))
    values = [x[1] for x in sorted_labels]
    return np.asarray(values)
