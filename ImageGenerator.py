import numpy as np
import random as rand
from scipy import ndimage
import matplotlib.image as mpimg
from matplotlib import transforms as tf
import os
import glob
from ImageModifier import * 

def GenerateImages(folder):
    
    adjustedAngle = 1.00
    path =  "dataset\\" + folder + "\\inputs\\"
    CleanFolder(path)
    for n in range(0, 4):
        for angle in range(0, (int)(95/adjustedAngle)):
            img = CreateImage()
            AddNoise(img)
            AddLine(img)
            img = Rotate(img, (angle+rand.uniform(0.75, 1.25)) * 3.7895 * adjustedAngle) # 360/95
            AddLine(img)
            img = Resize(img)
            img = CahngeContrast(img, rand.uniform(1.2, 1.35))
            Save(img, path + "img_" + str('{0:02d}'.format(angle)) + "_" + str(n) + ".png")
            
            
def CleanFolder(path):
    for file in glob.glob(path + "*.png"):
        os.remove(file)


def CreateImage():
    img = np.zeros((40, 40, 3)) # 30x30 for rotation
    return img


def AddNoise(img):
    for x in range(0, 40):
        for y in range(0,40):
            for c in range(0,3):
                img[y][x][c] = rand.uniform(0.2, 0.9)


def AddLine(img):
    for y in range(0, 21):
        ColorToWhite(img, 20, y)
    
    
def ColorToWhite(img, x, y):
    for c in range(0,3):
        img[y][x][c] = rand.uniform(0.98, 1)


def Rotate(img, degree):
    return ndimage.rotate(img, -degree, reshape=False, prefilter=False, mode='constant', order=1)


def Resize(img):
    y,x,c = img.shape
    startx = x//2-(20//2)
    starty = y//2-(20//2)    
    return img[starty:starty+20,startx:startx+20]

    
def Save(img, name):
    mpimg.imsave(name, img, format='png')


if __name__ == "__main__":
    print("Start")


    GenerateImages("train_auto-generated")
    
    print("End")
