import numpy as np


def modify(img):
    img = remove_small_values(img, 0.80)
    img = convert_to_grayscale(img)
    img = change_brightness(img, 0.71)
    img = change_contrast(img, 2.85)
    # img = remove_small_values(img, 0.975)
    img = sqrt(img)
    return img


def convert_to_grayscale(img):
    img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
    return img.reshape([20, 20, 1])


def change_contrast(img, factor):
    img = 0.5 + factor * (img - 0.5)
    img = np.clip(img, 0, 1)
    return img


def change_brightness(img, factor):
    img = factor * img
    img = np.clip(img, 0, 1)
    return img


def remove_small_values(img, theshold):
    for x in range(0, len(img)):
        for y in range(0, len(img[x])):
            for z in range(0, len(img[x][y])):
                if img[x][y][z] <= theshold:
                    for c in range(0, len(img[x][y])):
                        img[x][y][c] = 0
                    continue
    return img


def sqrt(img):
    for x in range(0, len(img)):
        for y in range(0, len(img[x])):
            for c in range(0, len(img[x][y])):
                img[x][y][c] = img[x][y][c] * img[x][y][c]
    return img
