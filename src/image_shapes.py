import sys
import os
import cv2
import numpy as np
# from skimage import io
import JSON_functions as JSONfunc
from matplotlib import pyplot as plt
import image_utilities as img_util


"""
tools for extracting shapes
"""

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
    #find the largest connectComponentMask to see
    return foreground_mask

def petal_shape_fromBB(petal_im,petal_filename,petal_image_path):
    '''
input: image file, then get JSON annotation information about rectangle to use in the grabcut algorithm
output: focused petal image (foreground) with everything else filtered out (background)
    '''
    img_name = 'F1P111_Spot_Side2_200730.JPG'
    img_location = '/home/rajt/Desktop/Pipeline_Dataset_Test'
    test_img = cv2.imread(os.path.join(img_location,img_name))
        
    new_dict = JSONfunc.get_annotations(img_name,img_location)
    
#    print('This is the petal_filename: ',petal_filename)
#    print('This is the head',head) 
    
    if new_dict['bounding_box']['name'] == 'rect':
        x1 = new_dict['bounding_box']['x']
        y1 = new_dict['bounding_box']['y']
        w1 = new_dict['bounding_box']['width'] 
        h1 = new_dict['bounding_box']['height']
    
    else:
        raise ValueError("No annotations for bounding box (BB) found in " + im + " located in " + file_routing)
    
    foreground_mask = get_petal_shape(test_img)
    
    #get rectangle parameters from annotations
    rect = (x1,y1,w1,h1)
    print(rect)
        
    whole_image = cv2.imread(petal_image_path)
    mask = np.zeros(whole_image.shape[:2],np.uint8)
    mask = foreground_mask
    print(whole_image.shape)
    
#    bgdModel = cv.bitwise_not(foreground_mask)
    bgdModel,fgdModel = np.zeros((1,65),np.float64),np.zeros((1,65),np.float64)
    
    mask, bgdModel, fgdModel = cv2.grabCut(whole_image,mask,rect,bgdModel,fgdModel,1,cv2.GC_INIT_WITH_MASK)
    print(type(fgdModel))
    #    cv2.imread(fgdModel,0)
#    cv2.imshow('Testing',fgdModel)

    mask2 = np.where((mask==0)|(mask==2),0,1).astype('uint8')
    whole_image = whole_image*mask2[:,:,np.newaxis]
    plt.imshow(whole_image),plt.colorbar(),plt.show()

    return fgdModel

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

    return im_th

def get_vein_shape(im):
    blur = cv2.GaussianBlur(im, (5, 5), 0)
    th, im_th = cv2.threshold(blur, 10, 255, cv2.THRESH_BINARY) # seperate light from dark look at threshold image later
    # possibly cropping out the straght line?
    
    label_count, labels, stats, centroids = cv2.connectedComponentsWithStats(im_th, 4, cv2.CV_32S)
    sizes = stats[:,-1]
    
    max_label = 1
    
    max_size = sizes[1]
    for i in range(2, label_count):
        if sizes[i] > max_size:
            max_label = i
            max_size = sizes[i]
    vein_label = np.array([max_label])
    petal = cv2.inRange(labels, vein_label, vein_label)
    return petal

#    img2 = np.zeros(labels.shape)
#    img2[labels == max_label] = 255
#    cv2.imshow("Biggest component", img2)
#    cv2.waitKey()
#    return label_count,labels
    #Assume that the petal is centered in the image
#    vein_label = np.array([labels[labels.shape[0]//2][labels.shape[1]//2]])
#    petal = cv2.inRange(labels, vein_label, vein_label)
#    return petal

def tobacco_analysis(image_filename,file_path):
    # im = cv2.imread(os.)
    croppedImg, new_dict = JSONfunc.img_crop(image_filename,file_path)
#    cv2.grabCut()
    return img


def main():
#    img_name = '/home/rajt/Desktop/Pipeline_Dataset_Test/F1P115_Vein_Side1_200802.jpg'
    img_name = '/home/rajt/Desktop/Ex_Tobacco_Data/5-20_leaf11_back.NEF'    
    im = cv2.imread(img_name,1)
#    print(im.shape)
    big_im = cv2.resize(im, (0,0), fx=2, fy=2)
    cv2.imshow('image',big_im)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
