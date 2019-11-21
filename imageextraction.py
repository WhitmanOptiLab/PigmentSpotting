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
    blur = cv2.blur(image,(5,5))
    edges = cv2.Canny(blur, 10 ,100)
    dilated = cv2.dilate(edges,kernel,iterations = 1)
    erosion = cv2.erode(dilated,kernel,iterations = 1)
    return erosion

def find_image_contours(image):
    contours, hierarchy =  cv2.findContours(image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    return contours

def extract_image_objects(cnts, image):
    original = image.copy()
    image_number = 0
    image_paths = []
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)
        ROI = original[y:y+h, x:x+w]
        cv2.imwrite("ROI_{}.png".format(image_number), ROI)
        image_paths.append("ROI_{}.png".format(image_number))
        image_number += 1
    return image, image_paths

def get_points(img_path):
    shape = cv2.imread(img_path)
    #grey_shape = cv2.cvtColor(shape, cv2.COLOR_BGR2GRAY)
    grey_shape = get_image_kmeans(shape, 2)
    grey_shape1 = process_image(grey_shape)
    o = cv2.ORB_create(100)
    points = o.detect(grey_shape1, None)
    shape_points = cv2.drawKeypoints(shape,points,color=(0,255,0), flags=0,outImage=np.array([]))
    return shape_points

def get_distance(im1, im2):
    #process images
    img1 = cv2.imread(im1)
    img2 = cv2.imread(im2)
    im1_k_means = get_image_kmeans(img1, 2)
    im2_k_means = get_image_kmeans(img2, 2)
    processed_im1 = process_image(im1_k_means)
    processed_im2 = process_image(im2_k_means)

    #get contours
    im1_contours = find_image_contours(processed_im1)
    im2_contours = find_image_contours(processed_im2)

    #translate contours around central moments
    im1_moments = []
    im2_moments = []

    for c in im1_contours:
        mom = cv2.moments(c)
        cX = int((mom["m10"])/mom["m00"])
        cY = int((mom["m01"])/mom["m00"])
        im1_moments.append([cX,cY])
    for c in im2_moments:
        mom = cv2.moments(c)
        cX = int((mom["m10"])/mom["m00"])
        cY = int((mom["m01"])/mom["m00"])
        im2_moments.append([cX,cY])

    print(im1_moments)


def main():
    image = cv2.imread(sys.argv[1])
    image_k_means = get_image_kmeans(image, sys.argv[2])
    image_edges = process_image(image_k_means)
    image_contours = find_image_contours(image_edges)
    image_objects, paths = extract_image_objects(image_contours, image)
    #extracted_image_points = get_points(paths[1])
    get_distance(paths[1], sys.argv[1])


if __name__ == '__main__':
    main()
