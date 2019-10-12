import cv2
import numpy as np
import sys
from skimage import io, color, filters, measure

# This is a pipeline to detect the petals in our images.
# The pipeline takes an image and a k-value(k-value of 2 is best) as input.
# The first steams is using a k-means clustering on the image.
# The image is then processed to find petal edges.
# The contours of the image are then found and then the pipeline returns the image with the petals identified.

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

def process_image(image):
    kernel = np.ones((5,5),np.uint8)
    original = image.copy()
    ret,thresh1 = cv2.threshold(image,127,255,cv2.THRESH_BINARY)
    kernel = np.ones((5,5),np.uint8)
    erosion = cv2.erode(thresh1,kernel,iterations = 1)
    #Removing noise from image
    blur = cv2.blur(image,(5,5))
    #finding edges using edge detection
    edges = cv2.Canny(blur, 10 ,100)
    #laplacian = cv2.Laplacian(image,cv2.CV_8UC1)
    #sobel = cv2.Sobel(laplacian,cv2.CV_8UC1, 0, 1, ksize=5)
    dilated = cv2.dilate(edges,kernel,iterations = 1)
    erosion = cv2.erode(dilated,kernel,iterations = 1)

    io.imshow(erosion)
    io.show()

    return erosion

def find_image_contours(image):
    contours, hierarchy =  cv2.findContours(image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    return contours

def extract_image_objects(cnts, image):
    original = image.copy()
    image_number = 0
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)
        ROI = original[y:y+h, x:x+w]
        #cv2.imwrite("ROI_{}.png".format(image_number), ROI)
        image_number += 1
    return image

def main():
    image = cv2.imread(sys.argv[1])
    image_k_means = get_image_kmeans(image, sys.argv[2])
    image_edges = process_image(image_k_means)
    image_contours = find_image_contours(image_edges)
    image_objects = extract_image_objects(image_contours, image)
    io.imshow(image_objects)
    io.show()


if __name__ == '__main__':
    main()
