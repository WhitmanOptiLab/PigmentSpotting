import os
import sys
import glob
import spot_detection
import vein_filtering
import image_alignment
import json
import JSON_functions as JSONfunc

import numpy as np

from cv2 import imshow, waitKey, imread, imwrite, IMREAD_GRAYSCALE


#function 1
'''
input: should only take input dataset
output: list of image pairs of vein and pigment (vein_petal_pairs)
'''
def parse_dataset(dataset_path):
    #pairing up like images
    dataset = os.listdir(dataset_path)
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

#function 2
'''
input: path to directory for new dataset and image pair list from function 1 (vein_petal_pairs)
output: 
'''
def process_dataset(dataset_path,new_dataset_path,vein_petal_pairs):
    
    for vein_petal_pair in vein_petal_pairs:
        #specific vein/petal image names parsed from pairs
        vein_image_name = vein_petal_pair[0]
        petal_image_name = vein_petal_pair[1]        
#        create full path for each image
        vein_image_path = os.path.join(dataset_path, vein_image_name)
        petal_image_path = os.path.join(dataset_path, petal_image_name)

        vein_image = imread(vein_image_path, IMREAD_GRAYSCALE)
        vein_dict = JSONfunc.get_annotations(vein_image_name,dataset_path)
        print('\nThis is our vein_dict:')
        print(json.dumps(vein_dict, indent=4, sort_keys=False))    
        
#        '/home/rajt/Desktop/Test_Data/F1P116_Vein_Center_210629.jpg'    
#        petal_image = imread('/home/rajt/Desktop/Test_Data/F1P116_Spot_Center_210629.JPG', IMREAD_GRAYSCALE)
#        imshow('test',petal_image)
#        waitKey(0)

        print('this is the petal_image_name: ',petal_image_name)
        print('this is the dataset_path: ',dataset_path)
        petal_image, petal_dict=JSONfunc.img_crop(petal_image_name,dataset_path)
        print('\nThis is our petal_dict:')
        print(json.dumps(petal_dict, indent=4, sort_keys=False))
#        petal_image = imread(petal_image_path,IMREAD_GRAYSCALE)
#        _, vein_aligned_image,_ = image_alignment.align_images(petal_image, petal_image, petal_image_name, petal_image_path)
        
        _, vein_aligned_image,_ = image_alignment.align_images(petal_image, vein_image)

        pca_image, spot_results = spot_detection.get_predictions(petal_image, "_", dump_to_file= False)
        vein_image_filtered = vein_filtering.vein_enhance(vein_image)
        vein_image_filtered = vein_filtering.vein_enhance(vein_aligned_image)*255 

        ext_idx = petal_image_name.rfind('.')
        imwrite(os.path.join(new_dataset_path, petal_image_name[:ext_idx] + "_PCA" + petal_image_name[ext_idx:]), pca_image)
        ext_idx = vein_image_name.rfind('.')
        imwrite(os.path.join(new_dataset_path, vein_image_name[:ext_idx] + "_Aligned" + vein_image_name[ext_idx:]),vein_aligned_image)
        imwrite(os.path.join(new_dataset_path, vein_image_name[:ext_idx] + "_Aligned" + "_Vein_Filtered" + vein_image_name[ext_idx:]), vein_image_filtered)
        spot_detection.save_to_file(spot_results, os.path.join(new_dataset_path, "GMM_Predictions" + petal_image_name))
        
'''
Terminal testing:
data: /home/rajt/Desktop/Pipeline_Dataset_Test
output: /home/rajt/Desktop/Pipeline_Output_Test
'''

def main():
    if len(sys.argv) < 3 : 
        raise ValueError("Usage: pipeline_script.py <Path to dataset> <Path to directory for proecessed data>")
#    process_dataset(sys.argv[1], sys.argv[2])
    dataset_path = sys.argv[1]

    vein_petal_pairs = parse_dataset(dataset_path)
    
#    #final send w/new_dataset_path and vein_petal_pairs
    new_dataset_path = sys.argv[2]
    
    process_dataset(dataset_path,new_dataset_path,vein_petal_pairs)
    
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