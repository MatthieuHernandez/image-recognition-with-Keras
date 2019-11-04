import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import glob
import operator
from operator import itemgetter
from Data import *
from ImageModifier import *

class Regression:

    @staticmethod
    def LoadSet(folders):
        inputs = []
        labels = []
        for folder in folders:
  
            if folder == "train":
                labels = np.concatenate((labels, Regression.__LoadJson(folder)), axis=None)
                
            path = "dataset\\" + folder + "\\inputs\\*.png"
            for filename in glob.glob(path):
                img = mpimg.imread(filename)[:,:,:3]
                
                data = Modify(img)
                #data = img.flatten(order='C')
                inputs.append(data)
                
                if folder != "train":
                    filename = filename.split('\\')[-1]
                    name = filename.split('_')[0]
                    value = filename.split('_')[1].split('.')[0]

                    label = np.zeros(100)
                    label[int(value)] = 1
                    labels = np.concatenate((labels, [label]), axis=None)

        return (np.asarray(inputs), np.asarray(labels))
    
    @staticmethod
    def __LoadJson(folder):
        labels = []
        path = "dataset\\" + folder + "\\labels.json"
        lines = LoadJson(folder)
        lines = sorted(lines.items(), key=operator.itemgetter(0))
        for line in lines :
            label = np.zeros(100)
            label[line[1]] = 1
            labels.append(label)
        labels = np.array(labels)
        return labels
        


if __name__ == "__main__":
    print("Start")
    
    trainSet = Regression.LoadSet(["train", "test_hard", "train_auto-generated"])

    print("End")
