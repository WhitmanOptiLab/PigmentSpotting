#!/usr/bin/env python
"""Python-Macduff: "the Macbeth ColorChecker finder", ported to Python.

Original C++ code: github.com/ryanfb/macduff/

Usage:
    # if pixel-width of color patches is unknown,
    $ python macduff.py examples/test.jpg result.png > result.csv

    # if pixel-width of color patches is known to be, e.g. 65,
    $ python macduff.py examples/test.jpg result.png 65 > result.csv
"""
from __future__ import print_function, division
import cv2 as cv
import numpy as np
from numpy.linalg import norm
from math import sqrt
from sys import stderr, argv
from copy import copy
import os
import NEF_utils
# from decimal import Decimal

_root = os.path.dirname(os.path.realpath(__file__))

# Each color square must takes up more than this percentage of the image
MIN_RELATIVE_SQUARE_SIZE = 0.0001

DEBUG = False
#DEBUG = True

MAX_CONTOUR_APPROX = 500  # default was 7


# pick the colorchecker values to use -- several options available in
# the `color_data` subdirectory
# Note: all options are explained in detail at
# http://www.babelcolor.com/colorchecker-2.htm
#color_data = os.path.join(_root, 'color_data',
#                          'xrite_passport_colors_sRGB-GMB-2005.csv')


#TODO: make path to color data an optional argument to find Macbeth function

def get_expected_colors(macbeth_width, macbeth_height, csv_name):

    color_data = os.path.join(_root, 'color_data',
                            csv_name)

    expected_colors = np.flip(np.loadtxt(color_data, delimiter=','), 1)
    expected_colors = expected_colors.reshape(macbeth_width, macbeth_height, 3)

    return expected_colors

# a class to simplify the translation from c++
class Box2D:
    """
    Note: The Python equivalent of `RotatedRect` and `Box2D` objects 
    are tuples, `((center_x, center_y), (w, h), rotation)`.
    Example:
    >>> cv.boxPoints(((0, 0), (2, 1), 0))
    array([[-1. ,  0.5],
           [-1. , -0.5],
           [ 1. , -0.5],
           [ 1. ,  0.5]], dtype=float32)
    >>> cv.boxPoints(((0, 0), (2, 1), 90))
    array([[-0.5, -1. ],
           [ 0.5, -1. ],
           [ 0.5,  1. ],
           [-0.5,  1. ]], dtype=float32)
    """
    def __init__(self, center=None, size=None, angle=0, rrect=None):
        if rrect is not None:
            center, size, angle = rrect

        # self.center = Point2D(*center)
        # self.size = Size(*size)
        self.center = center
        self.size = size
        self.angle = angle  # in degrees

    def rrect(self):
        return self.center, self.size, self.angle

def crop_patch(center, radius, image):
    """
    Returns a circular patch of sample area
    """
    px, py = center

    patchmask = cv.circle(np.zeros(image.shape, np.uint8), (int(px),int(py)), int(radius), (1,1,1), -1)
    patch = image[np.max(patchmask, axis=2).astype(np.bool_)]
    
    return patch


def contour_average(contour, image):
    """Assuming `contour` is a polygon, returns the mean color inside it.

    Note: This function is inefficiently implemented!!! 
    Maybe using drawing/fill functions would improve speed.
    """

    # find up-right bounding box
    xbb, ybb, wbb, hbb = cv.boundingRect(contour)

    # now found which points in bounding box are inside contour and sum
    def is_inside_contour(pt):
        return cv.pointPolygonTest(contour, pt, False) > 0

    from itertools import product as catesian_product
    from operator import add
    from functools import reduce
    bb = catesian_product(range(max(xbb, 0), min(xbb + wbb,  image.shape[1])),
                          range(max(ybb, 0), min(ybb + hbb,  image.shape[0])))
    pts_inside_of_contour = [xy for xy in bb if is_inside_contour(xy)]

    # pts_inside_of_contour = list(filter(is_inside_contour, bb))
    color_sum = reduce(add, (image[y, x] for x, y in pts_inside_of_contour))
    return color_sum / len(pts_inside_of_contour)


def rotate_box(box_corners):
    """NumPy equivalent of `[arr[i-1] for i in range(len(arr)]`"""
    return np.roll(box_corners, 1, 0)


def check_colorchecker(values, expected_values):
    """Find deviation of colorchecker `values` from expected values."""
    diff = (values - expected_values[:, :values.shape[1]]).ravel(order='K')
    diff = diff[~np.isnan(diff)]
    return sqrt(np.dot(diff, diff)) #/ sqrt(values.shape[1])

def draw_colorchecker(colors, centers, image, radius, expected_colors):
    image = np.copy(image)
    for observed_color, expected_color, pt in zip(colors.reshape(-1, 3),
                                                  expected_colors.reshape(-1, 3),
                                                  centers.reshape(-1, 2)):
        x, y = pt
        cv.circle(image, (int(x), int(y)), radius//2, expected_color.tolist(), -1)
        cv.circle(image, (int(x), int(y)), radius//4, observed_color.tolist(), -1)
        
    return image

class ColorChecker:
    def __init__(self, values, reference, error, points, size):
        self.error = error
        self.values = values 
        self.points = points
        self.reference = reference
        self.size = size
    def __str__(self):
        return "Color Checker: \n\terror:{error}, \n\tvalues:{values}, \n\treference={reference} \n\tlocations:{points}, \n\tsize:{size}".format(error=self.error, values=self.values, points=self.points, size=self.size, reference = self.reference)

def find_colorchecker(boxes, image, expected_colors, macbeth_width, macbeth_height, macbeth_squares,debug_filename=None, use_patch_std=True,
                      debug=DEBUG):

    points = np.array([[box.center[0], box.center[1]] for box in boxes]) 
    passport_box = cv.minAreaRect(points.astype('float32'))
    (x, y), (w, h), a = passport_box 
    box_corners = cv.boxPoints(passport_box)
    top_corners = sorted(enumerate(box_corners), key=lambda c: c[1][1])[:2]
    top_left_idx = min(top_corners, key=lambda c: c[1][0])[0]
    box_corners = np.roll(box_corners, -top_left_idx, 0)
    tl, tr, br, bl = box_corners

    if debug:
        debug_images = [copy(image), copy(image)]
        for box in boxes:
            pts_ = [cv.boxPoints(box.rrect()).astype(np.int32)]
            cv.polylines(debug_images[0], pts_, True, (255, 0, 0))
        pts_ = [box_corners.astype(np.int32)]
        cv.polylines(debug_images[0], pts_, True, (0, 0, 255))

        bgrp = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 0, 255)]
        for pt, c in zip(box_corners, bgrp):
            cv.circle(debug_images[0], tuple(np.array(pt, dtype='int')), 10, c)
        cv.imwrite(debug_filename, np.vstack(debug_images))

        print("Box:\n\tCenter: %f,%f\n\tSize: %f,%f\n\tAngle: %f\n" 
              "" % (x, y, w, h, a), file=stderr)
 
    landscape_orientation = True  # `passport_box` is wider than tall
    if norm(tr - tl) < norm(bl - tl): # Orientation: Determines if the rectangle is in landscape orientation (wider than tall) or portrait orientation.
        landscape_orientation = False
        
    #Calculate observed width (and observed height if defined)
    #find the min distance between box center points
    
    min_point_dist = []
    
    for point1 in points: # Minimum Distance Calculation: Computes the minimum distance between different points (box centers).
        for point2 in points:
            if (point1[0] != point2[0]) and (point1[1] != point2[1]):            
                min_point_dist.append(norm(point1-point2))
    min_box_dist = min(min_point_dist)
    #print('Min distance between 2 boxes: ',min_box_dist)

    average_size = int(sum(min(box.size) for box in boxes) / len(boxes))
    # Average Box Size: Calculates the average size of the boxes.

    if landscape_orientation:
        #calculate observed width as round(norm(box side length) / min distance)
        observed_width = round(norm(tr-tl)/min_box_dist)+1

        dx = (tr - tl)/(observed_width-1)

        if macbeth_height > 1:
            dy = (bl - tl)/(macbeth_height - 1)
            
        else:
            dy = 0
    else:
        observed_width = int(round(norm(bl-tl)/min_box_dist)+1)
        
        dx = (bl-tl)/(observed_width-1)
    
        if macbeth_height > 1:
            dy = (tr - tl)/(macbeth_height - 1)
        else:
            dy = 0

    if debug:
        print("Observed width: ", observed_width)

    # calculate the averages for our oriented colorchecker
    fuzzy_dims = (macbeth_height, 2*macbeth_width - observed_width)
    patch_values = np.empty(fuzzy_dims + (3,), dtype='float32')
    patch_points = np.empty(fuzzy_dims + (2,), dtype='float32')
    sum_of_patch_stds = np.array((0.0, 0.0, 0.0))
    in_bounds = (0, 2*macbeth_width - observed_width)
    for x in range(2*macbeth_width - observed_width):
        for y in reversed(range(macbeth_height)):
            center = tl + (x-(macbeth_width-observed_width))*dx + y*dy
            # print(f"center = {center}, tl = {tl}, macbeth_width = {macbeth_width}, observed_width = {observed_width}")
            px, py = center
            radius = (average_size - 3) / 2

            img_patch = crop_patch(center, radius, image)

            if not landscape_orientation:
                y = macbeth_height - 1 - y

            patch_points[y, x] = center
            # center point half average size from the edge of the image

            if ((px - radius) > 0) and ((px + radius) < image.shape[1])\
                and ((py - radius) > 0) and ((py + radius) < image.shape[0]):
                extracted_color = img_patch.mean(axis=0)
                patch_values[y, x] = extracted_color
                sum_of_patch_stds += img_patch.std(axis=(0, 1))

            elif x < macbeth_width - observed_width:
                in_bounds = (x+1, in_bounds[1])
            elif x >= macbeth_width:
                in_bounds = (in_bounds[0], x)
            else:

                raise Exception('Previously detected quad now appears to be out of bounds?!?!')

            if debug:
                
                center = (int(px), int(py))
                radius = int(average_size / 2)
                cv.circle(debug_images[1], center, radius, extracted_color, thickness=-1)
                cv.circle(debug_images[1], center, radius, (0,0,255), thickness=1)
    if debug:
        cv.imwrite(debug_filename, np.vstack(debug_images))
    # determine which orientation has lower error
    if in_bounds[0] == 0:
        orient_1_error = check_colorchecker(patch_values[:, :expected_colors.shape[1]], 
                                            expected_colors[:, :]) # orientation one: top left to top left
        
        orient_2_error = check_colorchecker(patch_values[::-1, expected_colors.shape[1]-1::-1],
                                            expected_colors[:, :]) # orientation two: top left to bottom right
    else:
        orient_1_error = float('inf')
        orient_2_error = float('inf')
    if in_bounds[1] == 2*macbeth_width - observed_width:
        orient_3_error = check_colorchecker(patch_values[:, -expected_colors.shape[1]:in_bounds[1]],
                                            expected_colors[:, :]) # orientation three: top left to top left shifted to left one
    
        orient_4_error = check_colorchecker(patch_values[::-1, -1:-(expected_colors.shape[1]+1):-1],
                                            expected_colors[:, :]) # orietation four: tope left to bottom right shifted right one
    else:
        orient_3_error = float('inf')
        orient_4_error = float('inf')

    if debug:
        print("error list: ", orient_1_error, orient_2_error, orient_3_error, orient_4_error)

    min_err = min(orient_1_error, orient_2_error, orient_3_error, orient_4_error)
    
    if min_err == orient_2_error or min_err == orient_4_error:  # rotate by 180 degrees if the rotated orientations have less error
        patch_values = patch_values[::-1, ::-1]
        patch_points = patch_points[::-1, ::-1]
        lr_err_diff = orient_4_error - orient_2_error
    else:
        lr_err_diff = orient_1_error - orient_3_error # this should  be less than zero if orientation one has less error than 3
    if lr_err_diff < 0: 
        # print(f"lr_err_diff proc: {lr_err_diff}")
        patch_values = patch_values[::, :expected_colors.shape[1]]
        patch_points = patch_points[::, :expected_colors.shape[1]]
    else:
        # print(f"error not proc")
        patch_values = patch_values[::, -expected_colors.shape[1]:]
        patch_points = patch_points[::, -expected_colors.shape[1]:]
    if use_patch_std:
        error = sum_of_patch_stds.mean() / macbeth_squares
    else:
        error = min(orient_1_error, orient_2_error)

    if debug:
        print("dx =", dx, file=stderr)
        print("dy =", dy, file=stderr)
        print("Average contained rect size is %d\n" % average_size, file=stderr)
        print("Orientation 1: %f\n" % orient_1_error, file=stderr)
        print("Orientation 2: %f\n" % orient_2_error, file=stderr)
        print("Error: %f\n" % error, file=stderr)

    return ColorChecker(error=error,
                        values=patch_values,
                        points=patch_points,
                        size=average_size,
                        reference=expected_colors)
    

def angle_cos(p0, p1, p2):
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return abs(np.dot(d1, d2) / np.sqrt(np.dot(d1, d1)*np.dot(d2, d2)))


def is_right_size(quad, patch_size, rtol=.25):
    """Determines if a (4-point) contour is approximately the right size."""
    cw = abs(np.linalg.norm(quad[0] - quad[1]) - patch_size) < rtol*patch_size
    ch = abs(np.linalg.norm(quad[0] - quad[3]) - patch_size) < rtol*patch_size
    return cw and ch


# stolen from icvGenerateQuads
def find_quad(src_contour, min_size, max_size=None, squariness=0.9, debug_image=None):

    for max_error in range(2, MAX_CONTOUR_APPROX + 1):
        dst_contour = cv.approxPolyDP(src_contour, max_error, closed=True)
        if len(dst_contour) == 4:
            break

        # we call this again on its own output, because sometimes
        # cvApproxPoly() does not simplify as much as it should.
        dst_contour = cv.approxPolyDP(dst_contour, max_error, closed=True)
        if len(dst_contour) == 4:
            break

    # reject non-quadrangles
    is_acceptable_quad = False
    is_quad = False
    if len(dst_contour) == 4 and cv.isContourConvex(dst_contour):
        is_quad = True
        avg_sidelen = cv.arcLength(dst_contour, closed=True) / 4.0
        area = cv.contourArea(dst_contour, oriented=False)

        diag1 = (dst_contour[0] - dst_contour[2])
        diag2 = (dst_contour[1] - dst_contour[3])
        center1 = (dst_contour[0] + dst_contour[2]) / 2.0
        center2 = (dst_contour[1] + dst_contour[3]) / 2.0
        centerdist = np.linalg.norm(center1-center2)
        d1 = np.linalg.norm(diag1)
        d2 = np.linalg.norm(diag2)

        # strattja.  Only accept those quadrangles which are more square
        # than rectangular and which are big enough
        cond = (d1*squariness <= d2 <= d1/squariness and
                np.abs(np.dot(diag1.flatten(), diag2.flatten()))/d1 < (1-squariness)*avg_sidelen and
                centerdist < (1-squariness)*avg_sidelen and
                min_size < area and ((not max_size) or area < max_size))

        if not cv.CALIB_CB_FILTER_QUADS or area > min_size and cond:
            is_acceptable_quad = True
            # return dst_contour

    if debug_image is not None:
        cv.drawContours(debug_image, [src_contour], -1, (255, 0, 0), 1)
        if is_acceptable_quad:
            cv.drawContours(debug_image, [dst_contour], -1, (0, 255, 0), 1)
        elif is_quad:
            cv.drawContours(debug_image, [dst_contour], -1, (0, 0, 255), 1)
        return debug_image

    if is_acceptable_quad:
        return dst_contour
    return None


def find_macbeth(macbeth_img, macbeth_width, macbeth_height, macbeth_reflectance_file , patch_size=None, is_passport=False, debug=DEBUG,
                 min_relative_square_size=MIN_RELATIVE_SQUARE_SIZE, ):
    
    macbeth_squares = macbeth_width * macbeth_height

    expected_colors = get_expected_colors(macbeth_height=macbeth_width , macbeth_width=macbeth_height, csv_name=macbeth_reflectance_file)

    macbeth_original = copy(macbeth_img)
    macbeth_split = cv.split(macbeth_img)

    # threshold each channel and OR results together
    block_size = int(min(macbeth_img.shape[:2]) * 0.10) | 1
    macbeth_split_thresh = []
    for channel in macbeth_split:
        _,res = cv.threshold(channel,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
        
#        res = cv.adaptiveThreshold(channel,
#                                    255,
#                                    cv.ADAPTIVE_THRESH_MEAN_C,
#                                    cv.THRESH_BINARY_INV,
#                                    block_size,
#                                    C=6)

        macbeth_split_thresh.append(res)
    adaptive = np.bitwise_and(*macbeth_split_thresh)

    if debug:
        print("Used %d as block size\n" % block_size, file=stderr)
        cv.imwrite('debug_threshold.png',
                    np.vstack(macbeth_split_thresh + [adaptive]))

    # do an opening on the threshold image
    element_size = int(2 + block_size / 10)
    shape, ksize = cv.MORPH_RECT, (element_size, element_size)
    element = cv.getStructuringElement(shape, ksize)
    adaptive = cv.morphologyEx(adaptive, cv.MORPH_OPEN, element)

    if debug:
        print("Used %d as element size\n" % element_size, file=stderr)
        cv.imwrite('debug_adaptive-open.png', adaptive)

    # find contours in the threshold image
    tmp = cv.findContours(image=adaptive,
                          mode=cv.RETR_LIST,
                          method=cv.CHAIN_APPROX_SIMPLE)
    
    try:
        contours, _ = tmp
    except ValueError:  # OpenCV < 4.0.0
        adaptive, contours, _ = tmp

    if debug:
        show_contours = cv.cvtColor(copy(adaptive), cv.COLOR_GRAY2BGR)
        cv.drawContours(show_contours, contours, -1, (0, 255, 0))
        cv.imwrite('debug_all_contours.png', show_contours)

    min_size = np.prod(macbeth_img.shape[:2]) * min_relative_square_size
    max_size = None
    if patch_size:
        min_size = max(min_size, (0.8*patch_size)**2)
        max_size = (1.2*patch_size)**2

    def is_seq_hole(c):
        return cv.contourArea(c, oriented=True) > 0

    def is_big_enough(contour):
        _, (w, h), _ = cv.minAreaRect(contour)
        return w * h >= min_size
    
    
#####   1 initial quad is being filtered out here, not sure if one of the 4
    # filter out contours that are too small or clockwise
    contours = [c for c in contours if is_big_enough(c) and is_seq_hole(c)]

    if debug:
        show_contours = cv.cvtColor(copy(adaptive), cv.COLOR_GRAY2BGR)
        cv.drawContours(show_contours, contours, -1, (0, 255, 0))
        cv.imwrite('debug_big_contours.png', show_contours)

        debug_img = cv.cvtColor(copy(adaptive), cv.COLOR_GRAY2BGR)
        for c in contours:
            debug_img = find_quad(c, min_size, debug_image=debug_img)
        cv.imwrite("debug_quads.png", debug_img)

    if contours:
        initial_quads = [find_quad(c, min_size, max_size) for c in contours]
        if is_passport and len(initial_quads) <= macbeth_squares:
            qs = [find_quad(c, min_size) for c in contours]
            qs = [x for x in qs if x is not None]
            initial_quads = [x for x in qs if is_right_size(x, patch_size)]
        initial_quads = [q for q in initial_quads if q is not None]
        initial_boxes = [Box2D(rrect=cv.minAreaRect(q)) for q in initial_quads]

        if debug:
            show_quads = cv.cvtColor(copy(adaptive), cv.COLOR_GRAY2BGR)
            cv.drawContours(show_quads, initial_quads, -1, (0, 255, 0))
            cv.imwrite('debug_quads2.png', show_quads)
            print("%d initial quads found", len(initial_quads), file=stderr)

        if is_passport or (len(initial_quads) > macbeth_squares):
            if debug:
                print(" (probably a Passport)\n", file=stderr)

            # set up the points sequence for cvKMeans2, using the box centers
            points = np.array([box.center for box in initial_boxes],
                              dtype='float32')

            # partition into two clusters: passport and colorchecker
            criteria = \
                (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            compactness, clusters, centers = \
                cv.kmeans(data=points,
                           K=2,
                           bestLabels=None,
                           criteria=criteria,
                           attempts=100,
                           flags=cv.KMEANS_RANDOM_CENTERS)

            partitioned_quads = [[], []]
            partitioned_boxes = [[], []]
            for i, cluster in enumerate(clusters.ravel()):
                partitioned_quads[cluster].append(initial_quads[i])
                partitioned_boxes[cluster].append(initial_boxes[i])

            debug_fns = [None, None]
            if debug:
                debug_fns = ['debug_passport_box_%s.jpg' % i for i in (0, 1)]

                # show clustering
                img_clusters = []
                for cl in partitioned_quads:
                    img_copy = copy(macbeth_original)
                    cv.drawContours(img_copy, cl, -1, (255, 0, 0))
                    img_clusters.append(img_copy)
                cv.imwrite('debug_clusters.jpg', np.vstack(img_clusters))

            # check each of the two partitioned sets for the best colorchecker
            partitioned_checkers = []
            for cluster_boxes, fn in zip(partitioned_boxes, debug_fns):
                partitioned_checkers.append(
                    find_colorchecker(cluster_boxes, macbeth_original, fn,
                                      debug=debug))

            # use the colorchecker with the lowest error
            found_colorchecker = min(partitioned_checkers,
                                     key=lambda checker: checker.error)

        else:  # just one colorchecker to test
            debug_img = None
            if debug:
                debug_img = "debug_passport_box.jpg"
                print("\n", file=stderr)

            found_colorchecker = \
                find_colorchecker(boxes= initial_boxes,
                                  image= macbeth_original,
                                  expected_colors= expected_colors,
                                  macbeth_height= macbeth_height, 
                                  macbeth_width= macbeth_width, 
                                  macbeth_squares= macbeth_squares, 
                                  debug_filename= debug_img,
                                  debug=debug)

        # render the found colorchecker
        image = draw_colorchecker(found_colorchecker.values,
                          found_colorchecker.points,
                          macbeth_img,
                          found_colorchecker.size,
                          expected_colors,
                          )

        #debugging circle placement
#        scale_percent = 10      
#        width = int(image.shape[1] * scale_percent / 100),height = int(image.shape[0] * scale_percent / 100)
#        dsize = (width, height)
#        output = cv.resize(image, dsize)
#        cv.imshow('example',output)
#        cv.waitKey(0)
        
        # print out the colorchecker info
        for color, pt in zip(found_colorchecker.values.reshape(-1, 3),
                             found_colorchecker.points.reshape(-1, 2)):
            b, g, r = color
            x, y = pt
            if debug:
                print("x: %.0f,y: %.0f,r: %.0f,g: %.0f,b: %.0f\n" % (x, y, r, g, b))
        if debug:
            print("This is the size of the Color Checker: %0.f\nThis is error: %f\n" 
                  "" % (found_colorchecker.size, found_colorchecker.error))
    else:
        raise Exception('Something went wrong -- no contours found')
    return macbeth_img, found_colorchecker


def write_results(colorchecker, filename=None):
    mes = ',r,g,b\n'
    for k, (b, g, r) in enumerate(colorchecker.values.reshape(1, 3)):
        mes += '{},{},{},{}\n'.format(k, r, g, b)

    if filename is None:
        print(mes)
    else:
        with open(filename, 'w+') as f:
            f.write(mes)


if __name__ == '__main__':
    if len(argv) == 3:
#        print('This is argv[1]: ',argv[1])
#        fragmented_list = os.path.splitext(argv[1])
#        print(fragmented_list[-1])
        out, colorchecker = find_macbeth(NEF_utils.generic_imread(argv[1]))
        cv.imwrite(argv[2], out)
    elif len(argv) == 4:
        out, colorchecker = find_macbeth(NEF_utils.generic_imread(argv[1]), patch_size=float(argv[3]))
        cv.imwrite(argv[2], out)
    else:
        print('Usage: %s <input_image> <output_image> <(optional) patch_size>\n'
              '' % argv[0], file=stderr)
#     write_results(colorchecker, 'results.csv')
