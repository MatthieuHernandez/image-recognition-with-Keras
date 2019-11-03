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
                
                #img = ConvertToGrayscale(img)
                #ConvertToGrayscale(img)
                data = Modify(img)
                #data = img.flatten(order='C')
                inputs.append(data)
                
                if folder != "train":
                    filename = filename.split('\\')[-1]
                    name = filename.split('_')[0]
                    value = filename.split('_')[1].split('.')[0]
                    if "fake" in folder:
                        label = float(value) / 1000.0
                    else:
                        label = float(value) / 100.0 #SpellCooldowns[name] WRONG
                    labels = np.concatenate((labels, [label]), axis=None)

        return (np.asarray(inputs), np.asarray(labels))
    
    @staticmethod
    def __LoadJson(folder):
        path = "dataset\\" + folder + "\\labels.json"
        labels = LoadJson(folder)
        for key in labels :
            name = key.split('_')[0]
            labels[key] = labels[key] / 100.0 #SpellCooldowns[name] WRONG
        sorted_labels = sorted(labels.items(), key=operator.itemgetter(0))
        values = [x[1] for x in sorted_labels]
        result = np.array(values)
        return result
        


if __name__ == "__main__":
    print("Start")
    
    trainSet = Regression.LoadSet(["train", "train_fake", "test_hard", "train_auto-generated"])

    print("End")
