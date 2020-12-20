"""
A program that datermines the redscale insensity of the pixels in an image.
Input: Sourcepath to a directory of images(in color), filepath to store output
Output: Numpy arrays converted into .csv files that contain the described data
On command line: python3 data_processing.py image_list filepath
"""


import sys
import principlecomponent as pc
import imghdr
import cv2 as cv
import os
from numpy import asarray
from numpy import savetxt

def get_file_names(pathname):
    return [f for f in os.listdir(pathname) if os.path.isfile(os.path.join(pathname, f))]

def store_data(filenames, input_path, output_path):
    filenum = 0
    for file in filenames:
        format_filename = input_path + '/' +file
        if imghdr.what(format_filename) == 'tiff':
            processing_message = "Processing " + file
            print(processing_message)
            im = cv.imread(format_filename)
            imdata = pc.pca_to_grey(im)
            filename = os.path.splitext(file)[0]
            new_filepath = output_path + filename + str(filenum)+ '.csv'
            savetxt(new_filepath, imdata, delimiter=',')
            complete_message = new_filepath + " data saved to " + output_path
            print(complete_message)
            filenum += 1
        else:
            error_message = "Error: incorrect file format " + im_file
            print(error_message)

def main():
    dir = sys.argv[1]
    input_filepaths = get_file_names(dir)
    output_filepath = sys.argv[2]
    store_data(input_filepaths, dir, output_filepath)



if __name__ == '__main__':
    main()
