import os
import sys
from cv2 import drawKeypoints
# import NEF_utils
import numpy as np
import json


def img_json_pairs(dir):
    """
    match all vein image and annotation file pairs
    """
    files = os.listdir(dir)
    veinans = list(filter(lambda x : ("vein" in x.lower()) and x.endswith(".json"), files))
    veinims = list(filter(lambda x : ("vein" in x.lower()) and x.lower().endswith(".jpg"), files))
    pairs = []
    for im in veinims:
        imname = im.lower().replace(".jpg","")
        for an in veinans:
            if imname == an.lower().replace("_labels","").replace("_label","").replace(".json",""):
                pairs.append((dir+"/"+im,dir+an))
    return pairs

def grab_all_data(dir):
    """ 
    get all vien images from the subdirectories of a directory
    """
    all_pairs = []
    for sub in os.listdir(dir):
        if os.path.isdir(dir + "/" + sub):
            subdir_pairs = img_json_pairs(dir + "/"+ sub)
            all_pairs = all_pairs + subdir_pairs
    return all_pairs




def precheck_annotation(path):
    """
    make sure that the annotation provided is in the standard format
    each annotation should be of length four, contain keys for each of the annotations, formatted correctly
    """
    data = json.load(open(path,))
    top_layer = list(data.keys())[0]
    if type(data[top_layer]['regions']) == dict:
        data_list = data[top_layer]['regions'].values()
    else:
        data_list = data[top_layer]['regions']

    if len(data_list) != 4:
        raise Exception(f"Malformed annotation, missing point: {path}")
    
    acceptable_labels = set(["center_vein_bottom", "center_vein_top", "left_cut_edge", "right_cut_edge"])
    
    for an in data_list:
        try:
            if an["region_attributes"]["label"].lower() not in acceptable_labels:
                raise Exception(f"Malformed annotation, mistyped label: {path}")
        except KeyError:
            raise Exception(f"Malformed annotation, missing label: {path}")

                





def parse_annotation(path):
    """
    parse a json annotation, returning only a list of the keypoints
    """
    data = json.load(open(path,))
    top_layer = list(data.keys())[0]
    new_dict = {}
    data_list = []
    annotation_list = []

    # handle old annotation type
    if type(data[top_layer]['regions']) == dict:
        data_list = data[top_layer]['regions'].values()
    else:
        data_list = data[top_layer]['regions']
    for an in data_list:
        if an["region_attributes"]["label"].lower() == "center_vein_bottom":
            annotation_list.append([an["shape_attributes"]["cx"],an["shape_attributes"]["cy"],1])
    for an in data_list:
        if an["region_attributes"]["label"].lower() == "center_vein_top":
            annotation_list.append([an["shape_attributes"]["cx"],an["shape_attributes"]["cy"],1])
    for an in data_list:
        if an["region_attributes"]["label"].lower() == "left_cut_edge":
            annotation_list.append([an["shape_attributes"]["cx"],an["shape_attributes"]["cy"],1])
    for an in data_list:
        if an["region_attributes"]["label"].lower() == "right_cut_edge":
            annotation_list.append([an["shape_attributes"]["cx"],an["shape_attributes"]["cy"],1])
    return np.array([annotation_list])

if __name__ == "__main__":
    
    precheck_annotation('/Users/oliverbaltzer/Google Drive/Shared drives/Cooley_Lab/Hybrid Speckling_Spot-Vein-Project/JoshuaShin_PetalPhotos//P3/F4P3_Vein_Side2_200806_labels.json')
    