import numpy as np

def Modify(img):
    img = RemoveSmallValues(img, 0.80)
    img = ConvertToGrayscale(img)
    #img = CahngeBrightness(img, 0.71)
    #img = CahngeContrast(img, 2.85)
    #img = RemoveSmallValues(img, 0.975)
    return img

def ConvertToGrayscale(img):
    img = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140]).reshape(20, 20, 1)
    return img

def CahngeContrast(img, factor):
    img = 0.5 + factor * (img - 0.5)
    img = np.clip(img, 0, 1)
    return img

def CahngeBrightness(img, factor):
    img = factor * img
    img = np.clip(img, 0, 1)
    return img

def RemoveSmallValues(img, factor):
    for x in range(0, len(img)):
        for y in range(0, len(img[x])):
            for z in range(0, len(img[x][y])):
                if img[x][y][z] <= factor:
                    for c in range(0, len(img[x][y])):
                        img[x][y][c] = 0
                    continue
    return img

