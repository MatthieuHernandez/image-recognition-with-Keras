import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import json
import glob
import operator
from operator import itemgetter
import Data

class Classification:

    @staticmethod
    def LoadSet(folder):
        data = Classification.LoadData(folder)
        labels = Classification.LoadLabels(folder)
        values =(data, labels)
        return values

    @staticmethod
    def LoadData(folder):
        set = []
        path = "Data\\Image\\dataset\\" + folder + "\\inputs\\*.png"
        for filename in glob.glob(path):
            img = mpimg.imread(filename)
            #data = img
            data = img.flatten(order='C')
            #data = img.reshape(20, 20, 3)
            set.append(data)
        return np.asarray(set)

    @staticmethod
    def LoadLabels(folder):
        path = "Data\\Image\\dataset\\" + folder + "\\labels.json"
        with open(path, 'r') as file:
            labels = json.load(file)
        for key in labels :
            name = key.split('_')[0]
            labels[key] = Data.SpellLabels[name]
            
        sorted_labels = sorted(labels.items(), key=operator.itemgetter(0))
        values = [x[1] for x in sorted_labels]
        return np.asarray(values)


if __name__ == "__main__":

    print("Start")

    trainSet = Classification.LoadSet("train")
    print(trainSet[1])
    
    print("End")
