import glob
import json
import operator
import os
from operator import itemgetter

import matplotlib.image as mpimg
import numpy as np

from data.image import modifier

currentDir = os.path.dirname(__file__)


class Regression:

    @staticmethod
    def load_set(folders):
        inputs = []
        labels = []
        for folder in folders:

            if folder == "train":
                labels = np.concatenate(
                    (labels, Regression.__load_json(folder)), axis=None)

            path = "Data\\Image\\dataset\\" + folder + "\\inputs\\*.png"
            for filename in glob.glob(path):
                img = mpimg.imread(filename)[:, :, :3]

                data = modifier.modify(img)
                inputs.append(data)

                if folder != "train":
                    filename = filename.split('\\')[-1]
                    #name = filename.split('_')[0]
                    value = filename.split('_')[1].split('.')[0]
                    if "fake" in folder:
                        label = float(value) / 1000.0
                    else:
                        # SpellCooldowns[name] WRONG
                        label = float(value) / 100.0
                    labels = np.concatenate((labels, [label]), axis=None)
        return (np.asarray(inputs), np.asarray(labels))

    @staticmethod
    def __load_json(folder):
        path = currentDir + "\\Image\\dataset\\" + folder + "\\labels.json"
        with open(path, 'r') as file:
            labels = json.load(file)
        for key in labels:
            labels[key] = labels[key] / 100.0  # SpellCooldowns[name] WRONG
        sorted_labels = sorted(labels.items(), key=operator.itemgetter(0))
        values = [x[1] for x in sorted_labels]
        result = np.array(values)
        return result


def main():
    print("Start")
    trainSet = Regression.load_set(
        ["train", "train_fake", "test_hard", "train_auto-generated"])
    print("End")


if __name__ == "__main__":
    main()
