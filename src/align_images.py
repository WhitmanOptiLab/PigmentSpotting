import cv2
import sys
from skimage import io
import image_alignment as align


def main():
    img1 = cv2.imread(sys.argv[1])
    img2 = cv2.imread(sys.argv[2], cv2.IMREAD_GRAYSCALE)
    dst = align.align_images(img1,img2, sys.argv[3])
    io.imshow(dst)
    io.show()



if __name__ == "__main__":
    main()
