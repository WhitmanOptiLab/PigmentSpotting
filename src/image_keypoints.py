import sys
import cv2
import numpy as np
import image_shapes as shapes
import principal_component as pca
from skimage import io
import math
import JSON_functions as JSONfunc
import image_utilities as img_util
import image_alignment as img_align
from os import path, listdir

def add_keypoints(petal_img, vein_img):
    #Adds keypoints to the petal image based on contours detected from the petal shape.
    #Adds keypoints along entire edge.
    petal_shape, vein_aligned, warp_matrix = img_align.align_images(petal_img, vein_img)
    #petal shape is a black and white mask of the shape. should be useful for keypoint detection
    contours, hierarchy = cv2.findContours(petal_shape, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    img_with_keypoints = petal_img.copy()
    counter = 0
    for cnt in contours:
        for point in cnt:
            if counter % 100 == 0: #module operator to reduce number of keypoints drawn
                cv2.circle(img_with_keypoints, tuple(point[0]), 10, (255,255,255), -1)
            counter += 1
    #things to add: points are in (x,y) format in json file as opposed to drawing them.
    return img_with_keypoints

def vein_axis(petal_img, vein_img, vein_annotation):
    #Creates the main vein axis based on the keypoints detected from the vein image.
    petal_shape, vein_aligned, warp_matrix = img_align.align_images(petal_img, vein_img)
    inv_warp_matrix = cv2.invertAffineTransform(warp_matrix) 
    vein_annotation_t = JSONfunc.get_transformed_annotations(vein_annotation,inv_warp_matrix)
    #extract vein top and bottom points from annotation
    vein_top = None
    vein_bottom = None
    for labelName in vein_annotation_t:
        if vein_annotation_t[labelName]['name'] == 'point':
            if 'vein__center_top' in labelName:
                vein_top = (vein_annotation_t[labelName]['cx'], vein_annotation_t[labelName]['cy'])
            if 'vein__center_bottom' in labelName:
                vein_bottom = (vein_annotation_t[labelName]['cx'], vein_annotation_t[labelName]['cy'])

    if vein_top is None or vein_bottom is None:
        raise ValueError("Vein top or bottom point not found in annotations.")
    #create vein axis vector
    vein_axis_vector = (vein_bottom[0] - vein_top[0], vein_bottom[1] - vein_top[1])
    vein_axis_length = math.sqrt(vein_axis_vector[0]**2 + vein_axis_vector[1]**2)
    vein_axis_unit_vector = (vein_axis_vector[0]/vein_axis_length, vein_axis_vector[1]/vein_axis_length)

    return vein_top, vein_bottom, vein_axis_unit_vector


def base_edge(petal_img, vein_img, vein_annotation):
    #Creates the base edge line based on the petal shape. Base egde is perpendicular to vein axis at the bottom point.
    #Base edge cuts the petal image so that the bottom edge is straight across while removing the least amount of petal area.
    #petal shape is the contour we need to extract keypoints from
    petal_shape, vein_aligned, warp_matrix = img_align.align_images(petal_img, vein_img)
    #find contours to extract the img from the background
    contours, hierarchy = cv2.findContours(petal_shape, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    main_contour = max(contours, key=cv2.contourArea)

    #get vein axis
    vein_top, vein_bottom, vein_axis_unit_vector = vein_axis(petal_img, vein_img, vein_annotation)
    #create perpendicular vector
    perp_vector = (-vein_axis_unit_vector[1], vein_axis_unit_vector[0])

    #create line through vein bottom point with perp_vector direction
    #find intersection points of this line with the petal contour
    line_len = 2000
    p0 = np.array([
        vein_bottom[0] - perp_vector[0] * line_len,
        vein_bottom[1] - perp_vector[1] * line_len
    ])
    p1 = np.array([
        vein_bottom[0] + perp_vector[0] * line_len,
        vein_bottom[1] + perp_vector[1] * line_len
    ])

    intersection_points = []

def linearFit():
    #based on points near to the bottom vein point, perform a linear fit to determine the base edge, then cut to fit
    pass

