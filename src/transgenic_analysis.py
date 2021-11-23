import os
import sys
import glob
import json
import macduff
import NEF_utils
import csv

import numpy as np
import cv2 as cv
from cv2 import imshow, waitKey, imread, imwrite, IMREAD_GRAYSCALE

DEBUG=False
SILENT=True

def get_annotations(image_filename, dataset_path):
    if image_filename[:-13] == '_labels.json':
        image_filename = json_filename
        
    else:
        base = os.path.splitext(image_filename)[0]
        json_filename = base + '_labels.json'
    
    json_full = os.path.join(dataset_path,json_filename)
    image_full = os.path.join(dataset_path,image_filename)
    data = json.load(open(json_full,))

    top_layer = [key for key in data.keys()][0]
    num_annotations = data[top_layer]['regions'].keys()
    
    new_dict = {}
    
    for region in num_annotations:
        attributes = data[top_layer]['regions'][region]['region_attributes']
        for attr_name in attributes:
            if attr_name.startswith('label'):
                labelName = attributes[attr_name]

        new_dict[labelName] = {}

    for region in num_annotations:
        shape = data[top_layer]['regions'][region]['shape_attributes']
        attributes = data[top_layer]['regions'][region]['region_attributes']
        for label in new_dict:
            if label == attributes['label']:
                new_dict[label][attributes["feature"].lower()] = shape

    return new_dict

def patch_analysis(pixelList):
    #Strength-of-green calculation
    return np.mean(pixelList[:,1] / np.sum(pixelList, axis=1))
 

def analyze_image(img, annotations):
    # For each group, generate patch and call processing

    results = {}

    for group in annotations:
        region = annotations[group]["analysis region"]
        points = np.array([a for a in zip(region["all_points_x"], region["all_points_y"])], np.int32).reshape((-1,1,2))
        points = points // 2 #NEF interlacing means JPEG annotations are double the coordinates we expect
        patchmask = cv.fillConvexPoly(np.zeros(img.shape[:2], np.int32), np.array(points), 1)

        InjectionSite = annotations[group]["injection site"]
        #Change this if you want the circle around the injection point to be bigger
        ExclusionRadius = 20
        patchmask = cv.circle(patchmask, (InjectionSite["cx"]//2, InjectionSite["cy"]//2), ExclusionRadius, 0, -1)
        maskedregion = img[np.array(patchmask, np.bool)]
        results[group] = patch_analysis(maskedregion)

    return results



def calibrate_image(rawimg, colorchecker):
    offset = []
    scale = []

    #Check for a linear fit with the colorchecker
    for axis in range(3):
        slope, intercept = np.polyfit(colorchecker.values[0,:,axis].flatten(), 
                                       colorchecker.reference[0,:,axis].flatten(),1)
        offset.append(intercept)
        scale.append(slope)

    offset = np.maximum(np.array(offset),0.0)
    scale = np.array(scale)
    img = rawimg[:,:] * scale + offset
    return img


'''
input: path to directory for new dataset and image pair list from output: 
'''
def process_dataset(dataset_path,image_list, outfile):

    results = []

    columns = []
    
    for imagefilename in image_list:
        print("Processing image " + imagefilename)
        #specific vein/petal image names parsed from pairs
#        create full path for each image
        image_path = os.path.join(dataset_path, imagefilename)

        image = imread(image_path)
        annotation_dict = get_annotations(imagefilename,dataset_path)

        if DEBUG:
            print('\nAnnotations for file ' + imagefilename + ' in ' + dataset_path + ':')
            print(json.dumps(annotation_dict, indent=4, sort_keys=False))    


        image = NEF_utils.generic_imread(image_path)

        # Find macbeth color chart
        try:
            macbeth_img, found_colorchecker = macduff.find_macbeth(image)
        except:
            print("Error finding colorchecker in image " + imagefilename)

            raise

        image = calibrate_image(image, found_colorchecker)

        analysis_results = analyze_image(image, annotation_dict)

        if len(columns) == 0:
            for label in analysis_results:
                columns.append(label)
            columns.sort()

        results.append([imagefilename] + [analysis_results[label] for label in columns])


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
        raise ValueError("Usage: transgenic_analysis.py <Path to dataset> <csv output filename>")
    dataset_path = sys.argv[1]
    filelist = os.listdir(dataset_path)
    if DEBUG:
        print(filelist)

    #Scan for .nef files with associated .json files to be processed
    imagelist = [x for x in filelist if x.lower().endswith("nef") and x[:-4]+'_labels.json' in filelist]

    if DEBUG:
        print(imagelist)
    
    process_dataset(dataset_path,imagelist, sys.argv[2])
    
if __name__ == "__main__":
   main()
    
#------------------------------------------------------------------------------------------
#OLD function
#def process_dataset(dataset_path, new_dataset_path):                
#               
#    for vein_petal_pair in vein_petal_pairs:
#        vein_image_name = vein_petal_pair[0]
#        vein_image = imread(os.path.join(dataset_path, vein_image_name), IMREAD_GRAYSCALE)
#        petal_image_na me = vein_petal_pair[1]
#        petal_image = imread(os.path.join(dataset_path, petal_image_name))
#        pimsz = np.shape(petal_image)
##        petal_image = petal_image[(7*pimsz[0])//24:(17*pimsz[0])//24,(7*pimsz[1])//24:(17*pimsz[1])//24]
#
#        #call image crop code
# 
#        _, vein_aligned_image = image_alignment.align_images(petal_image, vein_image)
#        pca_image, spot_results = spot_detection.get_predictions(petal_image, "_", dump_to_file= False)
#        #vein_image_filtered = vein_filtering.vein_enhance(vein_image)
#        vein_image_filtered = vein_filtering.vein_enhance(vein_aligned_image)*255
#        
#        ext_idx = petal_image_name.rfind('.')
#        imwrite(os.path.join(new_dataset_path, petal_image_name[:ext_idx] + "_PCA" + petal_image_name[ext_idx:]), pca_image)
#        ext_idx = vein_image_name.rfind('.')
#        imwrite(os.path.join(new_dataset_path, vein_image_name[:ext_idx] + "_Aligned" + vein_image_name[ext_idx:]), vein_aligned_image)
#        imwrite(os.path.join(new_dataset_path, vein_image_name[:ext_idx] + "_Aligned" + "_Vein_Filtered" + vein_image_name[ext_idx:]), vein_image_filtered)
#        spot_detection.save_to_file(spot_results, os.path.join(new_dataset_path, "GMM_Predictions" + petal_image_name))
