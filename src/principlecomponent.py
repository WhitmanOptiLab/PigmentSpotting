# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 15:04:44 2020

@authors: Abbey Felley, Jack Taylor
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np

def pca_to_grey(image):
    x,y,z = image.shape
    mat = image.reshape([x*y,z])

    mean, eigenvectors = cv2.PCACompute(mat, np.mean(mat, axis=0).reshape(1,-1))
    axis = eigenvectors[0,:].reshape([3])

    newmat = np.dot(mat, axis)
    newmat = np.around(newmat).astype(int)

    grey = newmat.reshape([x,y])
    return grey

def create_point_cloud(image):
    data = pca_to_grey(image)
    critical_points =[]

    for line in data:
        critical_points.append([0 if  d >= 100 else d for d in line])

    x = []
    y = []
    for i in range(len(critical_points)):
        for j in range(len(critical_points[i])):
            if critical_points[i][j] > 0:
                num_points = critical_points[i][j] // 10 # TODO: push to 10-25
                k = 0
                while(k < num_points):
                    x.append(j)
                    y.append(i)
                    k = k+1

            # else:
            #     x.append(j)
            #     y.append(i)

    X = np.array(x)
    Y = np.array(y)
    # plt.scatter(X,Y)
    # plt.invert_yaxis()
    # plt.show()
    return X,Y, data
