import sys

import cv2
import numpy as np
from skimage import io

import image_utilities as img_util

def get_image_kmeans(image, k):
    Z = image.reshape((-1,3))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = int(k)
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((image.shape))
    return res2

def process_image(im):
    im_in = cv2.bitwise_not(im)

    im_blurred = cv2.blur(im_in, (60,60))

    kernel = np.ones((2,2), np.uint8)

    im_closed = cv2.dilate(im_in, kernel, iterations=1)
    im_closed = cv2.erode(im_closed, kernel, iterations=1)

    im_closed = cv2.blur(im_closed, (20,20))
    return im_closed


def flood_fill(im):
    th, im_th = cv2.threshold(im, 220, 255, cv2.THRESH_BINARY_INV)
    im_floodfill = im_th.copy()
    h, w = im_th.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)

    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0,0), 255);

    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    im_out = im_th | im_floodfill_inv
    return im_out


def get_petal_shape(im):
    #Convert image to HSV color format for easier clustering
    im = cv2.GaussianBlur(im, (5, 5), 0)
    im = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
    kmeans = get_image_kmeans(im, 3)
    #Assume that the cluster of the top-right corner is the background cluster
    background = kmeans[0][0]
    foreground_mask = cv2.bitwise_not(cv2.inRange(kmeans, background, background))
    return foreground_mask


def get_filtered_vein_shape(im):
    im_in = cv2.bitwise_not(im)

    im_blurred = cv2.blur(im_in, (60,60))
    im_blurred = cv2.blur(im_blurred, (60,60))
    # im_blurred = cv2.blur(im_blurred, (60,60))

    kernel = np.ones((2,2), np.uint8)

    im_closed = cv2.dilate(im_blurred, kernel, iterations=1)
    im_closed = cv2.erode(im_closed, kernel, iterations=1)

    im_closed = cv2.blur(im_closed, (20,20))
    # Threshold.
    # Set values equal to or above 220 to 0.
    # Set values below 220 to 255.

    th, im_th = cv2.threshold(im_closed, 220, 255, cv2.THRESH_BINARY_INV)

    # Copy the thresholded image.
    im_floodfill = im_th.copy()

    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = im_th.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)

    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0,0), 255);

    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    # Combine the two images to get the foreground.
    im_out = im_th | im_floodfill_inv
    return im_th

def get_vein_shape(im):
    blur = cv2.GaussianBlur(im, (5, 5), 0)
    th, im_th = cv2.threshold(blur, 10, 255, cv2.THRESH_BINARY)
    label_count, labels = cv2.connectedComponents(im_th, 4, cv2.CV_32S)
    #Assume that the petal is centered in the image
    vein_label = np.array([labels[labels.shape[0]//2][labels.shape[1]//2]])
    petal = cv2.inRange(labels, vein_label, vein_label)
    return petal