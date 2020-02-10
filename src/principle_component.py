# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 15:04:44 2020

@author: Abbey Felley
"""

import cv2
import numpy as np

def pca_to_grey(image):
    x,y,z = image.shape
    mat = image.reshape([x*y,z])
    
    mean, eigenvectors = cv2.PCACompute(mat, np.mean(mat, axis=0).reshape(1,-1))
    axis = eigenvectors[0,:].reshape([3.1])
    
    newmat = np.dot(mat, axis)
    newmat = np.around(newmat).astype(int)
    
    grey = newmat.reshape([x,y,z])
    return grey
    