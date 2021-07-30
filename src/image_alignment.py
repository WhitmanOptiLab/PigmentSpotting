import sys
import cv2
import numpy as np
import image_shapes as shapes
import principal_component as pca
from skimage import io
import math
import JSON_functions as JSONfunc
import image_utilities as img_util

def shapeStatistics(shape_image):
    moments = cv2.moments(shape_image)
    center = moments["m10"]/moments["m00"], moments["m01"]/moments["m00"]
    area = moments["m00"]
    u20 = moments["mu20"]/moments["m00"]
    u02 = moments["mu02"]/moments["m00"]
    u11 = moments["mu11"]/moments["m00"]
    angle = (180 / np.pi) * 0.5 * np.arctan(2*u11 / (u20 - u02))

    #The angle given here will always be in the range [-45,45], whichever axis falls 
    # within that range. To make sure we get the major axis, we need use use this method:
    #  Citation: http://breckon.eu/toby/teaching/dip/opencv/SimpleImageAnalysisbyMoments.pdf
    if (u20 - u02) < 0:
        if u11 > 0:
            angle += 90
        else:
            angle -=90

    #Straighten the shape so that we can figure out which way is the "head" based on the aligned 
    # third moment (e.g. skewness)
    rotation = cv2.getRotationMatrix2D(center, -angle, 1).astype(np.float32)
    aligned = cv2.warpAffine(shape_image, rotation, shape_image.shape, flags=cv2.INTER_LINEAR)
#    io.imshow(aligned)
#    io.show()
    flat_moments = cv2.moments(aligned)
    if flat_moments["mu03"] < 0:
        angle += 180

    length = np.sqrt(2*(u20 + u02 + np.sqrt(4*(u11**2) + (u20-u02)**2)))
    width = np.sqrt(2*(u20 + u02 - np.sqrt(4*(u11**2) + (u20-u02)**2)))
    return (center, area, angle, length, width)

def match_images(petal_image, vein_image, s1, s2):
    sz = petal_image.shape
    #Consruct an initial guess of the transformation required to align the two images
    (PetalCenter, PetalArea, PetalAngle, PetalLength, PetalWidth) = shapeStatistics(s1)
    (VeinCenter, VeinArea, VeinAngle, VeinLength, VeinWidth) = shapeStatistics(s2)
    scale = math.sqrt(VeinArea/PetalArea)
    number_of_iterations = 100
    termination_eps = 1e-5
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)
    AngleDifference = VeinAngle - PetalAngle
    warp_matrix = cv2.getRotationMatrix2D(PetalCenter, AngleDifference, scale).astype(np.float32)
    warp_matrix[0][2] += VeinCenter[0]-PetalCenter[0]
    warp_matrix[1][2] += VeinCenter[1]-PetalCenter[1]
    #Getting annotations; recreate new dimensional conditions

#    print('vein_image: ',vein_image.shape)
        
#    test = cv2.warpAffine(cv2.bitwise_and(vein_image,s2), warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
#    io.imshow_collection([s1, test])
#    io.show()
    try:
        (cc, final_warp) = cv2.findTransformECC(s1,s2,warp_matrix, cv2.MOTION_AFFINE, criteria, inputMask=None, gaussFiltSize=5)
    except (cv2.error):
        cc = 0

    if cc < 0.9:
        print("Error: can't find satisfactory alignment, displaying masks for debug")
        io.imshow_collection([s1, s2])
        io.show()
    elif cc == 0:
        raise ValueError("Cannot find any alignment for the images provided.")
    return cv2.warpAffine(cv2.bitwise_and(vein_image,s2), final_warp, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP), final_warp

def combine_imgs(img1, img2):
    grimg = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    alpha = 0.5
    beta = (1.0 - alpha)
    dst = cv2.addWeighted(img1, alpha, grimg, beta, 0.0)
    return dst

#add arguements: petal_filename, petal_image_path,
def align_images(petal_img, vein_img, raw_vein=True):
    petal_shape = shapes.get_petal_shape(petal_img)
#    petal_shape = shapes.petal_shape_fromBB(petal_img,petal_filename,petal_image_path)
    if raw_vein: 
        vein_shape = shapes.get_vein_shape(vein_img)
    else:
        vein_shape = shapes.get_filtered_vein_shape(vein_img)
    #also take forward the 'warp_matrix' to use for annotations transformations
    vein_aligned,warp_matrix = match_images(petal_img, vein_img, petal_shape,vein_shape)
    return petal_shape, vein_aligned, warp_matrix

          
def main():
    
    if (len(sys.argv) < 3):
        raise ValueError("Usage: image_alignment.py <petal image> <vein image>")
    
    #petal initalization + annotation processing
    petal_image, new_dict = JSONfunc.img_crop(sys.argv[1])
    petal_x = new_dict["bounding_box"]['x']
    petal_y = new_dict["bounding_box"]['y']
    
    petal_warp_matrix = [[1,0,int(-petal_x)],[0,1,int(-petal_y)]]
    updated_dict = JSONfunc.get_transformed_annotations(new_dict,petal_warp_matrix)
    
    #vein initalization for image (vein_image) and dictionary (new_vein_dict)
    new_vein_dict = JSONfunc.get_annotations(sys.argv[2])
    vein_image = cv2.imread(sys.argv[2],0)
    #get 'warp_matrix' from 'align_images' function and set = to 'vein_warp_matrix'
    petal_shape, vein_aligned, warp_matrix = align_images(petal_img, vein_img, petal_filename, petal_image_path)
    #vein annotations
    inv_warp_matrix = cv2.invertAffineTransform(warp_matrix) 
    updated_vein_dict = JSONfunc.get_transformed_annotations(new_vein_dict,inv_warp_matrix)
    
    masked_petal = cv2.bitwise_and(petal_image,cv2.cvtColor(petal_shape, cv2.COLOR_GRAY2BGR),)    
    if (input("Show overlaid images? (y/n): ") == 'y'):    
        combined = combine_imgs(masked_petal, vein_aligned)
        combined = JSONfunc.display_annotations(updated_dict,combined)
        combined = JSONfunc.display_annotations(updated_vein_dict,combined)        #vein annotations
        inv_warp_matrix = cv2.invertAffineTransform(warp_matrix) 
        updated_vein_dict = JSONfunc.get_transformed_annotations(new_vein_dict,inv_warp_matrix)
        io.imshow(combined)
        io.show()
        
    petal_outfile = sys.argv[1][:sys.argv[1].rfind('.')] + "_pca" + sys.argv[1][sys.argv[1].rfind('.'):]
    cv2.imwrite(petal_outfile, pca.pca_to_grey(petal_image, petal_shape, True))
    vein_outfile = sys.argv[2][:sys.argv[2].rfind('.')] + "_aligned" + sys.argv[2][sys.argv[2].rfind('.'):]
    cv2.imwrite(vein_outfile, vein_aligned)

    
if __name__ == "__main__":
    main()