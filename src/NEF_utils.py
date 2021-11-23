import rawpy
import imageio
import matplotlib as mpl
import cv2 as cv
import PIL
import os

#path = '/home/rajt/pythonMacduff/examples/6-1-21_leaf14_back.NEF'
#rgb = cv.imread(path)
#print(rgb)
#raw = rawpy.imread(path)
#raw_image = raw.raw_image.copy()
#rgb = raw.postprocess()
##cv.imshow('example',rgb)
##cv.waitKey(0)

DEBUG=False

def NEF_processing(path):
    '''
input: image of .NEF format
output: file format in 32-bit integer RGB format parsable by the macduff.py code 
'''
    raw = rawpy.imread(path)
    
    rgb = raw.postprocess(half_size = True,gamma=(1,1), no_auto_bright=True, output_bps=8)

    if DEBUG:
        print('\nThis is the data type: ',rgb.dtype,'\nThis is the shape: ',rgb.shape)
    
    #scale_percent = 100
    #width = int(rgb.shape[1] * scale_percent / 100)
    #height = int(rgb.shape[0] * scale_percent / 100)
    #dsize = (width, height)
    #output = cv.resize(rgb, dsize)
    
    return rgb

def generic_imread(path):
    path_split = os.path.splitext(path)

    if path_split[-1].lower() == '.nef':
        return NEF_processing(path)
    else:
        return cv.imread(path)

    
if __name__ == "__main__":
    NEF_processing(sys.argv[1])
    
