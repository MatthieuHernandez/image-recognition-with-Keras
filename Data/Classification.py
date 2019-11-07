import glob
import json
import operator

import matplotlib.image as mpimg
import numpy as np

from data import information


class Classification:

    @staticmethod
    def load_set(folder):
        data = Classification.load_data(folder)
        labels = Classification.load_labels(folder)
        values = (data, labels)
        return values

    @staticmethod
    def load_data(folder):
        dataset = []
        path = "data\\image\\dataset\\" + folder + "\\inputs\\*.png"
        for filename in glob.glob(path):
            img = mpimg.imread(filename)
            data = img.flatten(order='C')
            dataset.append(data)
        return np.asarray(set)

    @staticmethod
    def load_labels(folder):
        path = "Data\\Image\\dataset\\" + folder + "\\labels.json"
        with open(path, 'r') as file:
            labels = json.load(file)
        for key in labels:
            name = key.split('_')[0]
            labels[key] = information.SPELL_LABELS[name]

        sorted_labels = sorted(labels.items(), key=operator.itemgetter(0))
        values = [x[1] for x in sorted_labels]
        return np.asarray(values)


def main():
    print("Start")
    train_set = Classification.load_set("train")
    print(train_set[1])
    print("End")


if __name__ == "__main__":
    main()
