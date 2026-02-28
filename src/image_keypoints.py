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
            if 'center_vein_top' in labelName:
                vein_top = (vein_annotation_t[labelName]['cx'], vein_annotation_t[labelName]['cy'])
            if 'center_vein_bottom' in labelName:
                vein_bottom = (vein_annotation_t[labelName]['cx'], vein_annotation_t[labelName]['cy'])

    if vein_top is None or vein_bottom is None:
        raise ValueError("Vein top or bottom point not found in annotations.")
    #create vein axis vector
    vein_axis_vector = (vein_bottom[0] - vein_top[0], vein_bottom[1] - vein_top[1])
    vein_axis_length = math.sqrt(vein_axis_vector[0]**2 + vein_axis_vector[1]**2)
    vein_axis_unit_vector = (vein_axis_vector[0]/vein_axis_length, vein_axis_vector[1]/vein_axis_length)

    return vein_top, vein_bottom, vein_axis_unit_vector


def line_segment_intersection(vein_bottom, perp_vector, P0, P1):
    """
    Solve B + t*v = P0 + u*(P1-P0)
    Returns (intersection_point, t, u) or (None, None, None)
    """
    vein_bottom = np.array(vein_bottom, dtype=float)
    perp_vector = np.array(perp_vector, dtype=float)
    P0 = np.array(P0, dtype=float)
    P1 = np.array(P1, dtype=float)

    seg = P1 - P0
    M = np.column_stack((perp_vector, -seg))  # [perp_vector | -(P1-P0)]

    # Check if matrix is invertible
    if abs(np.linalg.det(M)) < 1e-8:
        return None, None, None

    rhs = P0 - vein_bottom
    t, u = np.linalg.solve(M, rhs)

    if 0 <= u <= 1:  # intersection lies on the segment
        intersection = vein_bottom + t * perp_vector
        return intersection, t, u

    return None, None, None


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
    perp_vector = np.array([-vein_axis_unit_vector[1], vein_axis_unit_vector[0]])  # rotate by 90 degrees

    #create line through vein bottom point with perp_vector direction
    #find intersection points of this line with the petal contour
    intersection_points = []
    for i in range(len(main_contour)):
        P0 = main_contour[i][0]
        P1 = main_contour[(i+1) % len(main_contour)][0]  # next point (wrap around)
        intersection, t, u = line_segment_intersection(vein_bottom, perp_vector, P0, P1)
        if intersection is not None:
            intersection_points.append(intersection)

    if len(intersection_points) < 2:
        raise ValueError("Less than 2 intersection points found. Cannot determine base edge.")
    #choose the two intersection points that are farthest apart
    intersection_points = sorted(intersection_points, key=lambda p: p[0])
    left_pt, right_pt = intersection_points[0], intersection_points[-1]

    return left_pt, right_pt, perp_vector

def crop_below_base_line(vein_img, left_pt, right_pt, vein_bottom):
    # Unpack
    x1, y1 = left_pt
    x2, y2 = right_pt

    # Line coefficients
    A = y1 - y2
    B = x2 - x1
    C = x1*y2 - x2*y1

    # Determine which side is "below"
    test = A*vein_bottom[0] + B*vein_bottom[1] + C
    keep_positive = test > 0

    # Build mask
    h, w = vein_img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    # For each pixel, check which side of the line it's on
    yy, xx = np.mgrid[0:h, 0:w]
    side_vals = A*xx + B*yy + C

    if keep_positive:
        mask[side_vals >= 0] = 255
    else:
        mask[side_vals <= 0] = 255

    # Apply mask
    cropped = cv2.bitwise_and(vein_img, vein_img, mask=mask)

    # Optional: crop bounding box tightly
    ys, xs = np.where(mask > 0)
    if len(xs) > 0:
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        cropped = cropped[y_min:y_max+1, x_min:x_max+1]

    return cropped, mask

def draw_base_edge_line(petal_img, left_pt, right_pt, color=(0,0,255), thickness=3):
    p1 = tuple(left_pt.astype(int))
    p2 = tuple(right_pt.astype(int))
    img_with_line = petal_img.copy()
    cv2.line(img_with_line, p1, p2, color, thickness)
    return img_with_line

def perpendicular_line(petal_img, vein_img, vein_annotation):
    petal_shape, vein_aligned, warp_matrix = img_align.align_images(petal_img, vein_img)
    left_pt, right_pt, perp_vector = base_edge(petal_img, vein_img, vein_annotation)
    return draw_base_edge_line(vein_aligned, left_pt, right_pt)
  
def perpendicular_cut(petal_img, vein_img, vein_annotation):
    left_pt, right_pt, perp_vector = base_edge(petal_img, vein_img, vein_annotation)
    vein_top, vein_bottom, _ = vein_axis(petal_img, vein_img, vein_annotation)
    petal_shape, vein_aligned, warp_matrix = img_align.align_images(petal_img, vein_img)
    cropped_petal, mask = crop_below_base_line(vein_aligned, left_pt, right_pt, vein_bottom)
    return cropped_petal


def linearFit(petal_img, vein_img, vein_annotation):
    #based on points near to the bottom vein point, perform a linear fit to determine the base edge, then cut to fit
    vein_top, vein_bottom, vein_axis_unit_vector = vein_axis(petal_img, vein_img, vein_annotation)

    pass

