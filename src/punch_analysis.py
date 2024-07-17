import os
import sys
import glob
import json
import macduff
import NEF_utils
import csv
import colorcalibration

import numpy as np
import cv2 as cv
from cv2 import imshow, waitKey, imread, imwrite, IMREAD_GRAYSCALE

DEBUG=False
SILENT=False

def patch_analysis(pixelList):
    #Strength-of-green calculation for a list of pixels
    assert (len(np.shape(pixelList)) == 2) and (np.shape(pixelList)[1] == 3), "Internal Error: patch analysis argument invalid"
    return np.mean(pixelList[:,1] / np.sum(pixelList, axis=1))
 
def getBlobParams(patch_size):
    params = cv.SimpleBlobDetector_Params()
 
    # Filter by Area.
    params.filterByArea = True
    params.minArea = patch_size
    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.5

def analyze_image(img, patch_size=None):
    results = {}
    img_uint8 = np.uint8(img*2)
    detector = cv.SimpleBlobDetector_create(getBlobParams(patch_size))
    #SimpleBlobDetector only works on standard UINT8 images?
    keypoints = detector.detect(img_uint8)
    punch = max(keypoints, key=lambda x : x.size)
    # Show blobs
    if DEBUG:
        im_with_keypoints = cv.drawKeypoints(img_uint8, [punch], np.array([]), (0,0,255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv.imwrite("punch_keypoint.jpg", im_with_keypoints)
        print("Punch size: ", punch.size)

    patchmask = cv.circle(np.zeros(img.shape, np.int32), (int(punch.pt[0]), int(punch.pt[1])), int(punch.size/2 - 3), (1,1,1), -1)
    maskedregion = patchmask*img
    maskedregionpixellist = maskedregion[np.max(patchmask, axis=2).astype(np.bool_)]
    if DEBUG:
        imwrite("masked_region.png", maskedregion[:,:,::-1])
    assert np.sum(patchmask) // 3 == np.shape(maskedregionpixellist)[0], "Internal Error: inconsistent mask sizes"
    
    return patch_analysis(maskedregionpixellist)

'''
input: path to directory for new dataset and image pair list from output: 
'''
def process_dataset(dataset_path,image_list, outfile, patch_size=None, punch_size=None):

    results = []

    columns = ['Strength of Green']
    
    for imagefilename in image_list:
        print("Processing image " + imagefilename)
        #specific vein/petal image names parsed from pairs
#        create full path for each image
        image_path = os.path.join(dataset_path, imagefilename)

        image = NEF_utils.generic_imread(image_path)

        # Find macbeth color chart
        try:
            macbeth_img, found_colorchecker = macduff.find_macbeth(image, patch_size)
        except:
            print("Error finding colorchecker in image " + imagefilename)
            raise

        if DEBUG:
            print("Found color checker: \n", found_colorchecker)
        
        #Cheating - I know what the correct orientation is but didn't actually find it
        found_colorchecker.values = found_colorchecker.values[::-1, :]
        found_colorchecker.points = found_colorchecker.points[::-1, :]
        image = colorcalibration.calibrate_image(image, found_colorchecker)

        analysis_result = analyze_image(image, punch_size)
        results.append([imagefilename] + [analysis_result])


    columns = ["File"] + columns

    with open(outfile, 'w', newline='') as csvfile:
        filewriter = csv.writer(csvfile)
        filewriter.writerow(columns)
        for row in results:
            filewriter.writerow(row)

        
def main():
    global SILENT
    SILENT=False
    if len(sys.argv) < 3 : 
        raise ValueError("Usage: punch_analysis.py <Path to dataset> <csv output filename>")
    dataset_path = sys.argv[1]
    filelist = os.listdir(dataset_path)
    if DEBUG:
        print(filelist)

    #Scan for .nef files with associated .json files to be processed
    imagelist = [x for x in filelist if x.lower().endswith("nef") and x.lower().startswith("punch")]

    if DEBUG:
        print(imagelist)
    
    process_dataset(dataset_path,imagelist, sys.argv[2], 222, 1000)
    
if __name__ == "__main__":
   main()
