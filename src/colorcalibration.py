import numpy as np

def calibrate_image(rawimg, colorchecker):
    """calibrate_image(rawimg, colorchecker) -> img
    Calibrate the colors of an image using a detected color chart in that image.
    """
    offset = []
    scale = []

    #Check for a linear fit for each of the 3 color channels with the colorchecker
    for axis in range(3):
        
        validcells = np.isfinite(colorchecker.reference[0,:,axis].flatten())

        slope, intercept = np.polyfit(colorchecker.values[0,:,axis].flatten()[validcells], 
                                       colorchecker.reference[0,:,axis].flatten()[validcells],1)
        offset.append(intercept)
        scale.append(slope)

    offset = np.maximum(np.array(offset),0.0)
    scale = np.array(scale)
    img = rawimg[:,:] * scale + offset
    return img