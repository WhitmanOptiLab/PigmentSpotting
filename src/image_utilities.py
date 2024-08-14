import cv2
import numpy as np

def resize_image(img, size):
    """Resize an image.

    Args:
    img: The image to be resized.
    size: The target size.

    Returns:
        The resized image.
    """
    r = size / img.shape[1]
    dim = (size, int(img.shape[0] * r))
    return cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

def match_image_size(img1, img2):
    """Resize an image.

    Args:
    img1: The image with a target size.
    img2: The image to be resized.

    Returns:
        img2 resized to fit img1.
    """
    dim = (img1.shape[1])
    return cv2.resize(img2, dim, interpolation = cv2.INTER_AREA)

def brighten_image(img, alpha, beta):
    """Brighten an image.

    Args:
    img: The image to be brightened.
    alpha: The factor to brighten the image by.
    beta: A constant value to add to the brightened image.

    Returns:
        The brightened image.
    """
    new_image = (np.clip(alpha*(img.astype(np.int32)) + beta, 0, 255)).astype(img.dtype)
    return new_image

def make_bw(img):
    """Convert an image to black and white.

    Args:
    img: The image to be converted to black and white.

    Returns:
        The black and white image.
    """
    new_image = np.zeros(img.shape, img.dtype)
    new_image = np.clip(~img, 0, 1)*255
    return new_image

def remove_left_edge(img):
    """
    This function removes an edge present in many of the images, which causes shape detection to go awry if
    the petal touches the edge.

    Args:
    img: the full vein image in color

    Returns: image with the left edge cut out
    """

    dst = cv2.Canny(img, 50, 200, None, 3) # detect edges in image
    linesP = cv2.HoughLinesP(dst, 1, np.pi / 180, 100, 20, 100) # detect lines within image

    longest_line = None

    if linesP is not None:
        longest_line = max(linesP, key = \
                           lambda line : ((line[0][2] - line[0][0])**2 + (line[0][3] - line[0][1])**2)**(1/2))

    if longest_line is not None:
        pts = np.array([
                (0,0),
                (0,img.shape[0]),
                (longest_line[0][2], img.shape[0]),
                (longest_line[0][0], 0)
            ], np.int32)
            
        # Fill the polygon
        cv2.fillPoly(img, [pts], 0)
    
    return img