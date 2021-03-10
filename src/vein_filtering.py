import sys
import cv2 as cv
import numpy as np
from skimage import io, morphology
    
def vein_enhance(img):
    original = img
    # Converting the Image to Gray to optimize time: only working on one channel
    if len(np.shape(img)) > 2:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # First Filter: Applying a GaussianBlur to smooth the image and prepare it for processing
    img = cv.GaussianBlur(img,(19,19),0)
    # Second Filter: Using an adaptive threshold with generous operators  
    img = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,45,-7)
    # Third Filter: Close gaps betweenn veins with a Gaussian Blur 
    img = cv.GaussianBlur(img, (17,17), 0)
    # Fourth Filter: Using a global threshold
    ret, img = cv.threshold(img,130,255,cv.THRESH_BINARY) #Play with the first param to get best img
    # Fifth Filter: Using Connected Components with stats
    num_components, output_img, stats, centroids = cv.connectedComponentsWithStats(img, connectivity = 4)
    # Taking out the background component, so reducing the num_components  
    sizes, num_components = stats[1:, -1], num_components - 1
    # Setting up minimum size to eliminate small dots and noise 
    min_size = 50
    # Making the image empty, and then filling it only with the qualified components 
    img = np.zeros((output_img.shape))
    # for every component in the image, we wil keep it if it is above the minimum size
    for i in range(0, num_components):
        if sizes[i] >= min_size:
            img[output_img == i + 1] = 255
    # Sixth Filter: Applying Skeletonization after Binarizing the image
    img = img > 0
    img = morphology.skeletonize(img)  
    return img
       
if __name__ == "__main__":
    vein_image = cv.imread(sys.argv[1])
    # Converting the image to uint8 type
    result = vein_enhance(vein_image)*255
    # Saving the vein img after enhancing. 
    cv.imwrite("enhanced_vein_image.jpg", result)
    io.imshow(result > 0)
    io.show()