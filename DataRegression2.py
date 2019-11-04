import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import glob
import operator
from operator import itemgetter
from Data import *
from ImageModifier import *

class Regression2:

    @staticmethod
    def LoadSet(folders):
        inputs = []
        labels = []
        for folder in folders:
  
            if folder == "train":
                labels.append(Regression2.__LoadJson(folder))
                
            path = "dataset\\" + folder + "\\inputs\\*.png"
            for filename in glob.glob(path):
                img = mpimg.imread(filename)[:,:,:3]
                
                data = Modify(img)
                inputs.append(data)
                
                if folder != "train":
                    filename = filename.split('\\')[-1]
                    name = filename.split('_')[0]
                    value = filename.split('_')[1].split('.')[0]

                    label = np.zeros(100)
                    label[int(value)] = 1
                    labels.append(label)
                    a = np.asarray(inputs)
                    b = np.vstack(labels)
        return (a, b)
    
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
        return np.asarray(labels)
        


if __name__ == "__main__":
    print("Start")
    
    trainSet = Regression2.LoadSet(["train", "test_hard", "train_auto-generated"])

    print("End")
