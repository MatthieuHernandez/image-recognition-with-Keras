import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import glob
import operator
from operator import itemgetter
from Data import *

class Regression:

    @staticmethod
    def LoadSet(folder, folder2 = ""):
        data = Regression.LoadData(folder, folder2)
        labels = Regression.LoadLabels(folder, folder2)
        values =(data, labels)
        return values
    
    @staticmethod
    def LoadData(folder, folder2):
        set = []
        path = "dataset\\" + folder + "\\inputs\\*.png"
        for filename in glob.glob(path):
            img = mpimg.imread(filename)
            #img = Regression.__ConvertToGrayscale(img)
            #plt.imshow(img, cmap='gray')
            #plt.show()
            data = img.reshape(20, 20, 3)
            set.append(data)
        if folder2 != "":
            path = "dataset\\" + folder2 + "\\*.png"
            for filename in glob.glob(path):
                img = mpimg.imread(filename)
                data = img.reshape(20, 20, 3)
                set.append(data)
        return np.asarray(set)
    
    @staticmethod
    def LoadLabels(folder, folder2):
        labels = LoadJson(folder)       
        for key in labels :
            name = key.split('_')[0]
            labels[key] = labels[key] / SpellCooldowns[name]

        values2 = []  
        if folder2 != "":
            path = "dataset\\" + folder2 + "\\*.png"
            for filename in glob.glob(path):
                label = filename.split('\\')[-1].split('_')[1].split('.')[0]
                value = float(label)/100.0
                values2.append(value)
        sorted_labels = sorted(labels.items(), key=operator.itemgetter(0))
        values = [x[1] for x in sorted_labels]
        values = values + values2
        return np.asarray(values)
    
    def __ConvertToGrayscale(image):
        return np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])

    @staticmethod
    def __ImageTest():
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
    trainSet = Regression.LoadSet("train")
    print(trainSet)
    print("End")
