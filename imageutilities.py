import cv2
import numpy as np

def resize_image(img, size):
    r = size / img.shape[1]
    dim = (size, int(img.shape[0] * r))
    return cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

def match_image_size(img1, img2):
    dim = (img1.shape[1])
    return cv2.resize(img2, dim, interpolation = cv2.INTER_AREA)

def brighten_image(img, alpha, beta):
    new_image = np.zeros(img.shape, img.dtype)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            for c in range(img.shape[2]):
                new_image[y,x,c] = np.clip(alpha*img[y,x,c] + beta, 0, 255)
    return new_image

def make_bw(img):
    new_image = np.zeros(img.shape, img.dtype)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if img[y,x] == 255:
                new_image[y,x] =0
            else:
                new_image[y,x] = 255
    return new_image
