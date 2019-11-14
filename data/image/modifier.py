import numpy as np


def modify(img):
    img = reduce_colored_value(img, 0.135, 0.965)
    img = reduce_small_values(img, 0.80, 0.88)
    img = convert_to_grayscale(img)
    img = change_brightness(img, 0.84)
    img = change_contrast(img, 2.22)
    img = reduce_small_values(img, 0.8, 0.96)
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


def reduce_small_values(img, threshold, value):
    for x in range(0, len(img)):
        for y in range(0, len(img[x])):
            for z in range(0, len(img[x][y])):
                if img[x][y][z] <= threshold:
                    for c in range(0, len(img[x][y])):
                        img[x][y][c] = img[x][y][c] * value
                    continue
    return img


def reduce_colored_value(img, gap, value):
    for x in range(0, len(img)):
        for y in range(0, len(img[x])):
            if (abs(img[x][y][0] - img[x][y][1]) > gap or
                    abs(img[x][y][1] - img[x][y][2]) > gap or
                    abs(img[x][y][0] - img[x][y][2]) > gap):
                for c in range(0, 3):
                    img[x][y][c] = img[x][y][c] * value
                continue
    return img


def sqrt(img):
    for x in range(0, len(img)):
        for y in range(0, len(img[x])):
            for c in range(0, len(img[x][y])):
                img[x][y][c] = img[x][y][c] * img[x][y][c]
    return img
