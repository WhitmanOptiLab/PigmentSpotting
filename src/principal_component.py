import sys
import cv2
import os.path
import matplotlib.pyplot as plt
import numpy as np
import image_shapes as shapes
from skimage import io

def pca_to_grey(image, mask, inverted=False):
    x,y,z = image.shape
    mat = image.reshape([x*y,z])
    filter_array = mask.reshape([x*y])
    pointlist = mat[filter_array > 0]
    mean, eigenvectors = cv2.PCACompute(pointlist, mean=None)
    axis = eigenvectors[0,:].reshape([3])

    newmat = np.dot(mat.astype(np.float32) - mean, axis)
    newpoints = newmat[filter_array > 0]
    rescale = ((newmat - np.min(newpoints)) * 255.0 / (np.max(newpoints)-np.min(newpoints)))
    print(np.min(rescale), np.max(rescale))
    rescale = np.around(rescale).astype(np.uint8)

    grey = rescale.reshape([x,y])
    if inverted:
        grey = cv2.bitwise_not(grey)
    pigment = cv2.bitwise_and(grey, mask)
    return pigment

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

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Standalone usage: python principal_component.py input_filename.jpg [output_filename.jpg]")
        sys.exit(1)
    if not os.path.exists(sys.argv[1]): 
        raise ValueError("One of the image files doesn't Exists. Please review your file paths") 
    petal_image = cv2.imread(sys.argv[1])
    petal_shape = shapes.get_petal_shape(petal_image)
    result = pca_to_grey(petal_image, petal_shape, True)
    if len(sys.argv) > 2:
        cv2.imwrite(sys.argv[2], result)
    else:
        io.imshow(result)
        io.show()
