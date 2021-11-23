#JSON_functions.py
# has all functions that manipulate + work with JSON files and resulting data

import json
import cv2 as cv
import numpy as np
import os


def get_annotations(image_filename,routing):
    '''
get_annotations(image_filename) returns a dictionary with the annotations for a given image

input: image file name(type=str), and routing information to parent folder of file_name
output: return 'new dict' (dictionary of important information) 
'''
    print('image_filename: ',image_filename)
    
    if image_filename[:-13] == '_labels.json':
        image_filename = json_filename
        
    else:
        base = os.path.splitext(image_filename)[0]
#        print('Based Base: ',base)
        json_filename = base + '_labels.json'
    
    #elif:
    #support for processing without JSON files?
        
    print('json_file: ',json_filename)
    
    json_full = os.path.join(routing,json_filename)
    image_full = os.path.join(routing,image_filename)
#    print('json_full: ',json_full)
#    f = open(json_full,)
    data = json.load(open(json_full,))
    img = cv.imread(image_full,cv.IMREAD_COLOR)
    assert not isinstance(img,type(None)), 'image not found'
    
    top_layer = [key for key in data.keys()][0]
    num_annotations = data[top_layer]['regions'].keys()
    
    new_dict = {}
    
    print("Number of annotations: ",len(num_annotations))
    
    for key in num_annotations:
        print('We are on key number: ',key)
        keyInformation = data[top_layer]['regions'][key]['shape_attributes']
        attributes = data[top_layer]['regions'][key]['region_attributes']:

        for attr_name in attributes:
            if attr_name.startswith('label'):
                labelName = attributes[attr_name]

        print('Label name is: ',labelName)
        if labelName in new_dict:
            raise ValueError("Multiple labels of the same type not currently supported.\nFound duplicate label " + labelName + " in " + json_filename)     
        new_dict[labelName] = keyInformation                    
        
#    print(json.dumps(new_dict, indent=4, sort_keys=False))
    return new_dict

def organize_JSONfile(json_file):
    '''
organize_JSONfile(json_file) returns an organized dictionary for a given image

input: JSON file name 
output: returns organized JSON file 
'''
#    with open(json_file) as json_file: 
#        data = json.load(json_file)
#    print('\nFormatted .JSON is: ')
#    print(json.dumps(data, indent=4, sort_keys=False))
#    return data

    print('\nFormatted .JSON is: ')
    print(json.dumps(json_file, indent=4, sort_keys=False))
#    return data


def img_crop(image_filename,file_path):
    '''
img_crop(image)

input: image file name and directory leading to image folder
output: return croppedImg, new_dict
'''
    #create full file routing information
    image_full = os.path.join(file_path,image_filename)
    #read the image
    img = cv.imread(image_full,cv.IMREAD_COLOR)
    assert not isinstance(img,type(None)), 'image not found'
    
    #Get Truncated Dictionary:
    new_dict = get_annotations(image_filename,file_path)
    
    #Parse through each object's data for rectangle objects
    crop_key = 'bounding_box'
    if crop_key in new_dict:            
        #top left corner of rectangle (x1,y1) and width/height (w1,h1)
        x1,y1,w1,h1 = new_dict[crop_key]['x'],new_dict[crop_key]['y'],new_dict[crop_key]['width'],new_dict[crop_key]['height']

        #bottom right corner of rectangle (ending point)
        x2,y2 = int(x1+w1), int(y1+h1)
        
        croppedImg = img[(y1):(y2),(x1):(x2)]
    else:
        raise ValueError("Image crop failed: \n  Annotation key " + crop_key + " not found in annotations file for " + image)

    return croppedImg, new_dict


def get_transformed_annotations(input_dict,transformationMatrix):
    '''
input: pre-transformed values dictionary (input_dict) from 'get_annotations' function, transformationMatrix (2x3), and img 
output: return updated transfomration dictionary and image with annotations
'''
    #print('\nPre-Transformation Dictionary: ')    
    #print(json.dumps(input_dict, indent=4, sort_keys=False))
    updated_dict ={}
    
    for labelName in input_dict:
#        print(labelName)
        updated_dict[labelName] ={}        
        #should be in main()        
        #transformationMatrix = np.array([[1,0,-1231],[0,1,-1970]])
        #
        if input_dict[labelName]['name'] == 'center':
            input_dict[labelName]['name'] = 'point'
        
        if input_dict[labelName]['name'] == 'point':
            pointMatrix = np.array([[input_dict[labelName]['cx']],[input_dict[labelName]['cy']],[1]])
            resulting_pair = np.matmul(transformationMatrix, pointMatrix)
            #Update the dictionary with transformed coordinates
            updated_dict[labelName]['name'] = input_dict[labelName]['name']
            updated_dict[labelName]['cx'] = int(resulting_pair[0])
            updated_dict[labelName]['cy'] = int(resulting_pair[1])

        if input_dict[labelName]['name'] == 'rect':
            pointMatrix = np.array([ [input_dict[labelName]['x']],[input_dict[labelName]['y']],[1] ])
            resulting_pair = np.matmul(transformationMatrix, pointMatrix)
            #Update the dictionary with transformed coordinates
            updated_dict[labelName]['name'] = input_dict[labelName]['name']
            updated_dict[labelName]['x'] = int(resulting_pair[0])
            updated_dict[labelName]['y'] = int(resulting_pair[1])
            updated_dict[labelName]['width'] = input_dict[labelName]['width']
            updated_dict[labelName]['height'] = input_dict[labelName]['height']
                

    #print('Post-Transformation Dictionary: ')    
    #print(json.dumps(updated_dict, indent=4, sort_keys=False))
    return updated_dict  

def display_annotations(updated_dict,img):
    '''
Input: dictionary (updated_dict) and image file (img)
Output: returns both updated_dict and img 
'''
    for labelName in updated_dict:
        if updated_dict[labelName]['name'] == 'point':
            ptPair = (updated_dict[labelName]['cx'],updated_dict[labelName]['cy'])
            cv.circle(img,ptPair,5,(255,255,255),-1)

        if updated_dict[labelName]['name'] == 'rect':
            ptPair = (updated_dict[labelName]['x'],updated_dict[labelName]['y'])
            cv.rectangle(img,ptPair,5,(255,255,255),1)

    return img

def main():
    ''' image_filename
    Input: image filename (image_filename)
    Output: display image with annotations layer over 
    '''

#    img = cv.imread(image_filename,cv2.IMREAD_COLOR)
#    cv.imshow('Image loaded with annotations layer applied!',img)
#    cv.waitKey(0)
#
#    file_routing='/home/rajt/Desktop/Pipeline_Dataset_Test'
#    file_name='F1P111_Vein_Center_200731.jpg'
#    
#    new_dict = get_annotations(file_name,file_routing)
#    print(json.dumps(new_dict, indent=4, sort_keys=False))
    
#    croppedImg, new_dict = img_crop(file_name,file_routing)
#    print(new_dict)
#    print(json.dumps(new_dict, indent=4, sort_keys=False))
#
#    img = cv.imread(croppedImg)
#    cv.imshow('Testing:',croppedImg)
#    cv.waitKey(0)
            
#main()

