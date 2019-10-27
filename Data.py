import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import glob
import json
import operator
from operator import itemgetter

file = "test.png"

def LoadSet(folder):
    data = LoadData(folder)
    labels = LoadLabels(folder)
    values =(labels, data)
    return values

def LoadData(folder):
    set = []
    path = "dataset\\" + folder + "\\inputs\\*.png"
    for filename in glob.glob(path):
        img = mpimg.imread(filename)
        data = img.flatten(order='C')
        set.append(data)
    return np.asarray(set)

def LoadLabels(folder):
    path = "dataset\\" + folder + "\\labels.json"
    with open(path, 'r') as file:
        labels = json.load(file)
    sorted_labels = sorted(labels.items(), key=operator.itemgetter(0))
    values = [x[1] for x in sorted_labels]
    return np.asarray(values)

def ImageTest():
    img = mpimg.imread(file)
    print(img)
    print(img.shape)
    plt.imshow(img) # 20 *20 * 3
    imgFlat = img.flatten(order='C')
    print(imgFlat) # 1 * 1200
    plt.show()


if __name__ == "__main__":

    print("Start")
    
    #ImageTest()
    trainSet = LoadSet("train")
    #print(trainSet.shape)
    print("End")
