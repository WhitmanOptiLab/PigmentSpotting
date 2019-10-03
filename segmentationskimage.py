import numpy as np
import sys
from skimage import segmentation
from skimage import color
import matplotlib.pyplot as plt
from skimage import io
from skimage import filters
from skimage import data
from skimage import exposure
import cv2

def get_kmeans(image, K):
    img = cv2.imread(image)
    Z = img.reshape((-1,3))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret,label,center=cv2.kmeans(Z,int(K),None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    return res2

def img_to_greyscale(image):
    greyimg = color.rgb2grey(image)
    return greyimg

def get_edges_from_image(greyimg):
    edges = filters.sobel(greyimg)
    io.imshow(edges)
    io.show()

def otsu_thresholding(greyimg):
    val = filters.threshold_otsu(greyimg)
    hist, bins_center = exposure.histogram(greyimg)
    plt.figure(figsize=(9,4))
    plt.subplot(131)
    plt.imshow(greyimg, cmap="grey", interpolation="nearest")


def main():
    kmeans_img = get_kmeans(sys.argv[1], sys.argv[2])
    grey_img = img_to_greyscale(kmeans_img)
    get_edges_from_image(grey_img)

main()
