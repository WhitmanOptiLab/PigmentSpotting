import sys

import cv2
import numpy as np
import image_shapes as shapes
from skimage import io

import image_utilities as img_util

def match_images(petal_image, vein_image, s1, s2):
    sz = petal_image.shape
    #Consruct an initial guess of the transformation required to align the two images
    PetalMoments = cv2.moments(s1)
    PetalCenter = (PetalMoments["m10"]/PetalMoments["m00"], PetalMoments["m01"]/PetalMoments["m00"])
    VeinMoments = cv2.moments(s2)
    VeinCenter = (VeinMoments["m10"]/VeinMoments["m00"], VeinMoments["m01"]/VeinMoments["m00"])
    number_of_iterations = 300
    termination_eps = 1e-5
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)
    best_cc = 0
    best_transform = None
    for rotation in np.arange(0, 360, 90):
        warp_matrix = cv2.getRotationMatrix2D(PetalCenter, rotation, 1).astype(np.float32)
        warp_matrix[0][2] += VeinCenter[0]-PetalCenter[0]
        warp_matrix[1][2] += VeinCenter[1]-PetalCenter[1]
        try:
            (cc, final_warp) = cv2.findTransformECC(s1,s2,warp_matrix, cv2.MOTION_AFFINE, criteria)
        except (cv2.error):
            cc = 0
        if cc > best_cc:
            best_cc = cc
            best_transform = final_warp
        if best_cc > 0.99:
            break
        
    im2_aligned = cv2.warpAffine(cv2.bitwise_and(vein_image,s2), final_warp, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    if best_cc < 0.9:
        print("Error: can't find satisfactory alignment, displaying masks for debug")
        io.imshow_collection([s1, s2])
        io.show()
    return im2_aligned

    

def combine_imgs(img1, img2):
    grimg = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    alpha = 0.5
    beta = (1.0 - alpha)
    dst = cv2.addWeighted(img1, alpha, grimg, beta, 0.0)
    return dst

def align_images(petal_img, vein_img, raw_vein=True):
    petal_shape = shapes.get_petal_shape(petal_img)
    if raw_vein:
        vein_shape = shapes.get_vein_shape(vein_img)
    else:
        vein_shape = shapes.get_filtered_vein_shape(vein_img)

    vein_aligned = match_images(petal_img, vein_img, petal_shape,vein_shape)
    masked_petal = cv2.bitwise_and(petal_img, cv2.cvtColor(petal_shape, cv2.COLOR_GRAY2BGR))
    combined = combine_imgs(masked_petal, vein_aligned)
    return combined

def main():
    petal_image = cv2.imread(sys.argv[1])
    vein_image = cv2.imread(sys.argv[2], cv2.IMREAD_GRAYSCALE)
    dst = align_images(petal_image, vein_image)
    dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
    io.imshow(dst)
    io.show()

if __name__ == "__main__":
    main()
