import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib.image as mpimg

file = "test.png"

def imageTest():
    img = mpimg.imread(file)
    print(img)
    print(img.shape)
    plt.imshow(img) # 20 *20 * 3
    plt.show()


if __name__ == "__main__":

    print("Start")
    
    imageTest()

    print("End")
