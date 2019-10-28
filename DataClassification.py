import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import glob
import json
import operator
from operator import itemgetter

class Classification:

    SpellLabels ={
        "barrier":  0, # 180s
        "cleanse":  1, # 210s
        "exhaust":  2, # 210s
        "flash":    3, # 300s
        "ghost":    4, # 180s
        "heal":     5, # 240s
        "hexflash": 6, # 100s
        "ignite":   7, # 180s
        "smite":    8, # 100s # 100s at start and 90s after, big probelem game time needed
        "teleport": 9, # 360s 
    }

    def LoadSet(folder):
        data = Classification.LoadData(folder)
        labels = Classification.LoadLabels(folder)
        values =(data, labels)
        return values

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

    def LoadLabels(folder):
        path = "dataset\\" + folder + "\\labels.json"
        with open(path, 'r') as file:
            labels = json.load(file)  
        for key in labels :
            name = key.split('_')[0]
            labels[key] = Classification.SpellLabels[name]
        sorted_labels = sorted(labels.items(), key=operator.itemgetter(0))
        values = [x[1] for x in sorted_labels]
        return np.asarray(values)

if __name__ == "__main__":

    print("Start")

    trainSet = Classification.LoadSet("train")
    print(trainSet[1])
    
    print("End")
