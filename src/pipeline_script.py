import os
import sys
import glob
import spot_detection
import vein_filtering
import image_alignment

import numpy as np

from cv2 import imread, imwrite, IMREAD_GRAYSCALE

def process_dataset(dataset_path, new_dataset_path):

    dataset = os.listdir(dataset_path)
    fileNameComponents = []
    for file_name in dataset:
        fileNameComponents.append(file_name.split('_'))
    vein_petal_pairs = []
    for file_name_1 in fileNameComponents:
        for file_name_2 in fileNameComponents:
            if (file_name_1[0] == file_name_2[0]) and (file_name_1[1] != file_name_2[1]) and (file_name_1[2] == file_name_2[2]):
                if (file_name_1[1] == 'Vein'):
                    vein_petal_pair = ('_'.join(file_name_1), '_'.join(file_name_2))
                else:
                    vein_petal_pair = ('_'.join(file_name_2), '_'.join(file_name_1))
                vein_petal_pairs.append(vein_petal_pair)

    for vein_petal_pair in vein_petal_pairs:
        vein_image_path = os.path.join(dataset_path, vein_petal_pair[0])
        vein_image = imread(vein_image_path, IMREAD_GRAYSCALE)
        petal_image_path = os.path.join(dataset_path, vein_petal_pair[1])
        petal_image = imread(petal_image_path)
        _, vein_aligned_image = image_alignment.align_images(petal_image, vein_image)
        pca_image, spot_results = spot_detection.get_predictions(petal_image, "_", dump_to_file= False)
        #vein_image_filtered = vein_filtering.vein_enhance(vein_image)
        vein_image_filtered = vein_filtering.vein_enhance(vein_aligned_image)
        with open(new_dataset_path) as dir:
            imwrite((petal_image_path[:petal_image_path.rfind('.')] + "_PCA" + petal_image_path[petal_image_path.rfind('.'):]), pca_image)
            imwrite((vein_image_path[:vein_image_path.rfind('.')] + "_Aligned" + vein_image_path[vein_image_path.rfind('.'):]), vein_aligned_image)
            imwrite((vein_image_path[:vein_image_path.rfind('.')] + "_Vein_Filtered" + vein_image_path[vein_image_path.rfind('.'):]), vein_image_filtered)
            spot_detection.save_to_file(spot_results, "GMM_Predictions".join(petal_image_path))


def main():
    if len(sys.argv) < 3 : 
        raise ValueError("Usage: pipeline_script.py <Path to dataset> <Path to directory for proecessed data>")
    process_dataset(sys.argv[1], sys.argv[2])

if __name__ == "__main__":
   main()
    

    
