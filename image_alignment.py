import cv2
import numpy as np
import imageutilities as img_util
import image_shapes as shapes

def match_images(im1,im2,s1,s2):
    sz = im1.shape
    warp_mode = cv2.MOTION_EUCLIDEAN

    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else:
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    number_of_iterations = 30000

    termination_eps = 1e-5

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)

    (cc, warp_matrix) = cv2.findTransformECC(s1,s2,warp_matrix, warp_mode, criteria, None, 1)

    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        im2_aligned = cv2.warpPerspective(im2, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else:
        im2_aligned = cv2.warpAffine(im2, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

    return im2_aligned

def combine_imgs(img1, img2):
    grimg = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    alpha = 0.8
    beta = (1.0 - alpha)
    dst = cv2.addWeighted(grimg, alpha, img2, beta, 0.0)
    return dst

def align_images(petal, vein, size):
    petal_img = img_util.resize_image(petal, 500)
    vein_img = img_util.resize_image(vein, int(size))

    petal_shape = shapes.get_petal_shape(petal_img)
    vein_shape = shapes.get_vein_shape(vein_img)

    vein_aligned = match_images(petal_img, vein_img, petal_shape,vein_shape)
    return combine_imgs(petal_img, vein_aligned)
