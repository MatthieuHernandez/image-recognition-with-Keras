import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import glob
import operator
from operator import itemgetter
from Data import *

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
        path = "dataset\\" + folder + "\\inputs\\*.png"
        for filename in glob.glob(path):
            img = mpimg.imread(filename)
            #data = img
            data = img.flatten(order='C')
            #data = img.reshape(20, 20, 3)
            set.append(data)
        return np.asarray(set)

    @staticmethod
    def LoadLabels(folder):
        labels = LoadJson(folder) 
        for key in labels :
            name = key.split('_')[0]
            labels[key] = SpellLabels[name]
            
        sorted_labels = sorted(labels.items(), key=operator.itemgetter(0))
        values = [x[1] for x in sorted_labels]
        return np.asarray(values)


if __name__ == "__main__":

    print("Start")

    trainSet = Classification.LoadSet("train")
    print(trainSet[1])
    
    print("End")
