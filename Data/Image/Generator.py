import glob
import os
import random as rand

import matplotlib.image as mpimg
import numpy as np
from scipy import ndimage

from data.image import modifier


def generate_images(folder):

    path = "Data\\Image\\dataset\\" + folder + "\\inputs\\"
    clean_folder(path)
    for _group in range(0, 2):
        for angle in range(0, 100):
            img = create_image()
            add_noise(img)
            add_noise(img)
            img = rotate(img, (angle+1) * rand.uniform(3.5856, 3.6144))
            add_line(img)
            img = resize(img)
            img = modifier.change_contrast(img, rand.uniform(1.6, 1.7))
            save(img, path + "img_" +
                 str('{0:02d}'.format(angle)) + "_" + str(_group) + ".png")


def clean_folder(path):
    for file in glob.glob(path + "*.png"):
        os.remove(file)


def create_image():
    img = np.zeros((40, 40, 3))  # 30x30 for rotation
    return img


def add_noise(img):
    for x in range(0, 40):
        for y in range(0, 40):
            for c in range(0, 3):
                img[y][x][c] = rand.uniform(0.2, 0.8)


def add_line(img):
    for y in range(0, 21):
        color_to_white(img, 20, y)


def color_to_white(img, x, y):
    for c in range(0, 3):
        img[y][x][c] = rand.uniform(0.90, 1)


def rotate(img, degree):
    return ndimage.rotate(img, -degree, reshape=False, prefilter=False, mode='constant', order=1)


def resize(img):
    y, x = img.shape
    startx = x//2-(20//2)
    starty = y//2-(20//2)
    return img[starty:starty+20, startx:startx+20]


def save(img, name):
    mpimg.imsave(name, img, format='png')


def main():
    print("Start")
    generate_images("train_auto-generated")
    print("End")


if __name__ == "__main__":
    main()
