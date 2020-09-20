import sys

import cv2
import numpy as np
from skimage import io
    
def vein_enhance(img):
    img = cv2.blur(img, (3,3))
    io.imshow(img)
    io.show()
    img = cv2.adaptiveThreshold(img, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,25,-15)
    return img
    
if __name__ == "__main__":
    petal_image = cv2.imread(sys.argv[1])
    result = vein_enhance(vein_image)
    io.imshow(result)
    io.show()
