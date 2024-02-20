import numpy as np
import cv2

# sharpening
def sharpen(img):
    kernel3 = np.array([[0, -1,  0],
                        [-1,  5, -1],
                        [0, -1,  0]])
    new_img = cv2.filter2D(src=img, ddepth=-1, kernel=kernel3)
    return new_img

# Unsharp masking
def unsharp_masking55(img):
    kernel12 = -(1 / 256.0) * np.array([[1, 4, 6, 4, 1],
                                        [4, 16, 24, 16, 4],
                                        [6, 24, -476, 24, 6],
                                        [4, 16, 24, 16, 4],
                                        [1, 4, 6, 4, 1]])


    new_img = cv2.filter2D(src=img, ddepth=-1, kernel=kernel12)
    return new_img

# Blur
def gaussianBlur(img):
    kernel10 = (1 / 16.0) * np.array([[1, 2, 1],
                                      [2, 4, 2],
                                      [1, 2, 1]])

    new_img = cv2.filter2D(src=img, ddepth=-1, kernel=kernel10)
    return new_img

