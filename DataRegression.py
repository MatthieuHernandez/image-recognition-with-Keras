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
    def LoadSet(folders):
        inputs = []
        labels = []
        for folder in folders:
  
            if folder == "train":
                #labels = labels +
                LoadJson(folder)
                
            path = "dataset\\" + folder + "\\inputs\\*.png"
            for filename in glob.glob(path):
                img = mpimg.imread(filename)
                img = Regression.__ConvertToGrayscale(img)

                #plt.figure('window title')
                plt.imshow(img, cmap='gray')##

                data = img.reshape(20, 20, 1)
                inputs.append(data)
                
                if folder != "train":
                    filename = filename.split('\\')[-1]
                    name = filename.split('_')[0]
                    value = filename.split('_')[1].split('.')[0]
                    if "fake" in folder:
                        label = float(value)/1000.0
                    else:
                        label = float(value) / SpellCooldowns[name]
                    labels.append(label)

                plt.show()

        return (np.asarray(inputs), np.asarray(labels))
    
    @staticmethod
    def LoadJson(folder):
        print("a")
        labels = LoadJson("dataset\\" + folder + "\\labels.json")       
        for key in labels :
            name = key.split('_')[0]
            labels[key] = labels[key] / SpellCooldowns[name]
        sorted_labels = sorted(labels.items(), key=operator.itemgetter(0))
        values = [x[1] for x in sorted_labels]
        print("b")
        return values
    
        '''values2 = []  
        if folder2 != "":
            path = "dataset\\" + folder2 + "\\*.png"
            for filename in glob.glob(path):
                label = filename.split('\\')[-1].split('_')[1].split('.')[0]
                value = float(label)/1000.0
                values2.append(value)
        sorted_labels = sorted(labels.items(), key=operator.itemgetter(0))
        
        values = values2 + values
        #if folder == "train":###########
        #    values = values2
        return np.asarray(values)'''
    
    def __ConvertToGrayscale(image):
        return np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])

    @staticmethod
    def __ImageTest():
        img = mpimg.imread("train_fake//inputs//black125.png")
        #print(img)
        #print(img.shape)
        plt.imshow(img) # 20 *20 * 3
        imgFlat = img.flatten(order='C')
        #print(imgFlat) # 1 * 1200
        #plt.show()

showData
for earch
image[0]
label as title

if __name__ == "__main__":

    print("Start")  
    #ImageTest()
    trainSet = Regression.LoadSet(["train", "train_fake", "test_hard"])
    showData(trainSet)
    print(trainSet)
    print("End")
