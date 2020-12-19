import principal_component as p
import sys
import os.path
import cv2
from skimage import io


def main():
    if not os.path.exists(sys.argv[1]): 
        raise ValueError("The image files doesn't Exists. Please review your file path")
    image = cv2.imread(sys.argv[1])
    im = vein_image = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
    i = p.pca_to_grey(image)
    io.imshow(im)
    io.show()
    io.imshow(i, cmap='gray', vmin=0, vmax=255)
    io.show()


main()
