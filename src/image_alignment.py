import sys
import cv2
import numpy as np
import image_shapes as shapes
import principal_component as pca
from skimage import io
import math
import JSON_functions as JSONfunc
import image_utilities as img_util
import CPUProfiler
from os import path, listdir

def shapeStatistics(shape_image):
    """
    skew identified in different directions
    orientation could use improvement
    """
    with CPUProfiler.CPUProfiler("shapeStatistics"):
        moments = cv2.moments(shape_image)
        center = moments["m10"]/moments["m00"], moments["m01"]/moments["m00"] # centeroid formula
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
        flat_moments = cv2.moments(aligned)
        if flat_moments["mu03"] < 0:
            angle += 180

        length = np.sqrt(2*(u20 + u02 + np.sqrt(4*(u11**2) + (u20-u02)**2)))
        width = np.sqrt(2*(u20 + u02 - np.sqrt(4*(u11**2) + (u20-u02)**2)))
        return (center, area, angle, length, width)

def match_images(petal_image, vein_image, s1, s2):
    with CPUProfiler.CPUProfiler("match_images"):
        petal_image_cpu = petal_image.get() 
        sz = petal_image_cpu.shape
        #Consruct an initial guess of the transformation required to align the two images
        (PetalCenter, PetalArea, PetalAngle, PetalLength, PetalWidth) = shapeStatistics(s1)
        # print(f"petal stats: center = {PetalCenter}, area = {PetalArea}, angle = {PetalAngle}")
        (VeinCenter, VeinArea, VeinAngle, VeinLength, VeinWidth) = shapeStatistics(s2)
        # print(f"vein stats: center = {VeinCenter}, area = {VeinArea}, angle = {VeinAngle}")
        scale = math.sqrt(VeinArea/PetalArea)
        number_of_iterations = 100
        termination_eps = 1e-5
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)
        AngleDifference = VeinAngle - PetalAngle
        # warp matrix with the vanilla angle difference
        warp_matrix = cv2.getRotationMatrix2D(PetalCenter, AngleDifference, scale).astype(np.float32)
        # warp matrix with the 180 deg rotated image
        warp_matrix_rotated = cv2.getRotationMatrix2D(PetalCenter, AngleDifference + 180, scale).astype(np.float32)
        warp_matrix[0][2] += VeinCenter[0] - PetalCenter[0] # translation offset
        warp_matrix[1][2] += VeinCenter[1] - PetalCenter[1]

        warp_matrix_rotated[0][2] += VeinCenter[0] - PetalCenter[0] # translation offset
        warp_matrix_rotated[1][2] += VeinCenter[1] - PetalCenter[1]
        #Getting annotations; recreate new dimensional conditions

        try:
            with CPUProfiler.CPUProfiler("findTransformECC"):
                (cc, warp) = cv2.findTransformECC(s1,s2,warp_matrix, cv2.MOTION_AFFINE, criteria, inputMask=None, gaussFiltSize=5)
            with CPUProfiler.CPUProfiler("findTransformECC_rotated"):
                (cc0, warp0) = cv2.findTransformECC(s1, s2, warp_matrix_rotated, cv2.MOTION_AFFINE, criteria, inputMask=None, gaussFiltSize=5)
            
            # print(f"cc: {cc} cc1: {cc0}")

            if cc0 > cc:
                cc = cc0
                warp = warp0

        except (cv2.error):
            cc = 0

        if cc < 0.9:
            print("Error: can't find satisfactory alignment, displaying masks for debug")
            io.imshow_collection([s1, s2])
            io.show()
        elif cc == 0:
            raise ValueError("Cannot find any alignment for the images provided.")
        return cv2.warpAffine(cv2.bitwise_and(vein_image,s2), warp, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP), warp

def combine_imgs(img1, img2):
    grimg = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    alpha = 0.5
    beta = (1.0 - alpha)
    dst = cv2.addWeighted(img1, alpha, grimg, beta, 0.0)
    return dst

#add arguements: petal_filename, petal_image_path,
def align_images(petal_img, vein_img, raw_vein=True):
    with CPUProfiler.CPUProfiler("align_images"):
        petal_shape = shapes.get_petal_shape(petal_img)
        #petal_shape = shapes.petal_shape_fromBB(petal_img,petal_filename,petal_image_path)
        if raw_vein: 
            vein_shape = shapes.get_vein_shape(vein_img)
        else:
            vein_shape = shapes.get_filtered_vein_shape(vein_img)
        #also take forward the 'warp_matrix' to use for annotations transformations
        #convert to Umat for GPU acceleration
        vein_img = cv2.UMat(vein_img)
        vein_aligned,warp_matrix = match_images(petal_img, vein_img, petal_shape,vein_shape) 
        # vein shape and petal shape are the masks of each petal 
    
    return petal_shape, vein_aligned, warp_matrix

def add_keypoints(petal_img, vein_img):
    petal_shape, vein_aligned, warp_matrix = align_images(petal_img, vein_img)
    #petal shape is a black and white mask of the shape. should be useful for keypoint detection
    contours, hierarchy = cv2.findContours(petal_shape, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    petal_img_cpu = petal_img.get() #convert Umat back to numpy array for drawing
    img_with_keypoints = petal_img_cpu.copy()
    counter = 0
    for cnt in contours:
        for point in cnt:
            if counter % 100 == 0: #module operator to reduce number of keypoints drawn
                cv2.circle(img_with_keypoints, tuple(point[0]), 10, (255,255,255), -1)
            counter += 1
    #things to add: points are in (x,y) format in json file as opposed to drawing them.
    return img_with_keypoints
    
          
def get_file_pairs(dir):
    dataset = listdir(dir)
    fileNameComponents = []
    for file_name in dataset:
        fileNameComponents.append(file_name.split('_'))
    vein_petal_pairs = []
    for file_name_1 in fileNameComponents:
        for file_name_2 in fileNameComponents:
            if (file_name_1[0] == file_name_2[0]) and (file_name_1[1] != file_name_2[1]) and (file_name_1[2] == file_name_2[2] and len(file_name_1) < 5 and len(file_name_2) < 5):
                if (file_name_1[1] == 'Vein'):
                    vein_petal_pair = ('_'.join(file_name_1), '_'.join(file_name_2))
                    vein_petal_pairs.append(vein_petal_pair)
    return vein_petal_pairs

    


def main():
    if cv2.ocl.haveOpenCL():
        print("OpenCL is available.")
        # Enable OpenCL use
        cv2.ocl.setUseOpenCL(True)
        print("OpenCL is enabled:", cv2.ocl.useOpenCL())
    else:
        print("OpenCL is not available or supported on this system.")
    with CPUProfiler.CPUProfiler("main"):
    
        if (len(sys.argv) != 2):
            raise ValueError("Usage: image_alignment.py <image_directory>")
        
        image_directory = sys.argv[1]

        image_pairs = get_file_pairs(image_directory) #takes a while sometimes

        show = input("Show overlaid images? (y/n): ")

        for pair in image_pairs:

            if "vein" in pair[0].lower():
                vein_img_filename = pair[0]
                petal_img_filename = pair[1]
            else:
                vein_img_filename = pair[1]
                petal_img_filename = pair[0]

            petal_image, petal_annotation = JSONfunc.img_crop(petal_img_filename, image_directory)
            
            petal_x = petal_annotation["bounding_box"]["x"]
            petal_y = petal_annotation["bounding_box"]["y"]


            petal_warp_matrix = [[1,0,int(-petal_x)],[0,1,int(-petal_y)]] # adjust the petal annotation

            petal_annotation_t = JSONfunc.get_transformed_annotations(petal_annotation, petal_warp_matrix)

            #vein initalization for image (vein_image) and dictionary (new_vein_dict)

            vein_annotation = JSONfunc.parse_annotation(vein_img_filename, image_directory, group_attr="label")
            vein_image = cv2.imread(path.join(image_directory, vein_img_filename),0)
            vein_image = cv2.UMat(vein_image) #testing gpu acceleration
            petal_image = cv2.UMat(petal_image) #testing gpu acceleration

            #get 'warp_matrix' from 'align_images' function and set = to 'vein_warp_matrix'

            petal_shape, vein_aligned, warp_matrix = align_images(petal_image, vein_image)
            img_with_keypoints = add_keypoints(petal_image, vein_image)
            if show == "y":
                io.imshow(img_with_keypoints)
                io.show()
                io.imshow(petal_shape)
                io.show()
                io.imshow(petal_image) 
                io.show()
                io.imshow(vein_aligned)
                io.show()
            #cv2.imshow('petal shape',petal_shape)
            

            inv_warp_matrix = cv2.invertAffineTransform(warp_matrix) 
            
            vein_annotation_t = JSONfunc.get_transformed_annotations(vein_annotation,inv_warp_matrix)

            #vein annotations
            #believe this to be the cause of the incorrect colors however removing it makes the images different sizes?
            masked_petal = cv2.bitwise_and(petal_image,cv2.cvtColor(petal_shape, cv2.COLOR_GRAY2BGR),) 
            #io.imshow(masked_petal)
            #io.show()
            
            combined = combine_imgs(masked_petal, vein_aligned)
            combined = JSONfunc.display_annotations(petal_annotation_t,combined)
            #io.imshow(combined) #unnessecary?
            combined = JSONfunc.display_annotations(vein_annotation_t,combined)        #vein annotations
            inv_warp_matrix = cv2.invertAffineTransform(warp_matrix) 

            if (show == "y"):    
                
                # updated_vein_dict = JSONfunc.get_transformed_annotations(vein_annotation,inv_warp_matrix)
                io.imshow(combined)
                #color images of saved images
                io.show()
            
            combined_outfile = petal_img_filename[:petal_img_filename.rfind('.')] + "_combined" + petal_img_filename[petal_img_filename.rfind('.'):]
            combined_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
            cv2.imwrite(combined_outfile, combined_bgr)
            petal_outfile = petal_img_filename[:petal_img_filename.rfind('.')] + "_pca" + petal_img_filename[petal_img_filename.rfind('.'):]
            cv2.imwrite(petal_outfile, pca.pca_to_grey(petal_image, petal_shape, True))
            vein_outfile = vein_img_filename[:vein_img_filename.rfind('.')] + "_aligned" + vein_img_filename[vein_img_filename.rfind('.'):]
            cv2.imwrite(vein_outfile, vein_aligned)
        

    
if __name__ == "__main__":
    main()