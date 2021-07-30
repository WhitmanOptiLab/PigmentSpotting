import rawpy
import imageio
import matplotlib as mpl
import cv2 as cv
import PIL

#path = '/home/rajt/pythonMacduff/examples/6-1-21_leaf14_back.NEF'
#rgb = cv.imread(path)
#print(rgb)
#raw = rawpy.imread(path)
#raw_image = raw.raw_image.copy()
#rgb = raw.postprocess()
##cv.imshow('example',rgb)
##cv.waitKey(0)

def NEF_processing(path):
    '''
input: image of .NEF format
output: file format parsable by the macduff.py code 
'''
    raw = rawpy.imread(path)
#    rgb = raw.postprocess(use_camera_wb=True)
#    rgb = raw.raw_image_visible.copy().astype('uint8')
    
    rgb = raw.postprocess(half_size = True,gamma=(1,1), no_auto_bright=True, output_bps=8)

    print('\nThis is the data type: ',rgb.dtype,'\nThis is the shape: ',rgb.shape)
    
    scale_percent = 100
    width = int(rgb.shape[1] * scale_percent / 100)
    height = int(rgb.shape[0] * scale_percent / 100)
    dsize = (width, height)
    output = cv.resize(rgb, dsize)
    
    return output
#    cv2.imwrite('D:/cv2-resize-image-50.png',output) 
#    
#    cv.imshow('rgb',output)
#    cv.waitKey(0)
#    raw.close()
    
def main(): 
    path = '/home/rajt/pythonMacduff/examples/6-1-21_leaf14_back.NEF'
    NEF_processing(path)
    
main()
    