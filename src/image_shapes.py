
import cv2
import sys
import numpy as np
from skimage import io
import imageutilities as img_util

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

    b_im = img_util.brighten_image(im, 2,0)
    kmeans = get_image_kmeans(b_im, 3)
    im_in =  cv2.cvtColor(kmeans,cv2.COLOR_BGR2GRAY)
    im_in = img_util.make_bw(im_in)
    im_closed = process_image(im_in)
    return flood_fill(im_closed)


def get_vein_shape(im):
    im_closed = process_image(im_in)
    th, im_th = cv2.threshold(im_closed, 220, 255, cv2.THRESH_BINARY_INV)
    return im_th
