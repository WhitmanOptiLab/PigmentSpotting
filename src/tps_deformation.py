import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage.transform import ThinPlateSplineTransform, warp

# Example image
image = data.checkerboard()

# Source keypoints (landmarks) -> need to specify points
src = np.array([[22, 22], [100, 10], [177, 22], [190, 100],
                [177, 177], [100, 188], [22, 177], [10, 100]])

# Target keypoints (desired warped positions) ->need to specify points
dst = np.array([[0, 0], [100, 0], [200, 0], [200, 100],
                [200, 200], [100, 200], [0, 200], [0, 100]])

# Create TPS transform and estimate from target to source
tps = ThinPlateSplineTransform()
tps.estimate(dst, src)  # Note inverse mapping for warp()

# Warp the image using the TPS transform
warped = warp(image, tps)

# Display original and warped images with landmarks
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(image, cmap='gray')
ax1.scatter(src[:, 0], src[:, 1], marker='x', color='red')
ax1.set_title('Original Image')

ax2.imshow(warped, cmap='gray', extent=(0, 200, 200, 0))
ax2.scatter(dst[:, 0], dst[:, 1], marker='x', color='red')
ax2.set_title('Warped Image')

plt.show()
# import numpy
# # Insight Toolkit (ITK) repository, which demonstrates deforming a 3D volume using TPS with source and target landmarks
# import itk 
# import argparse

# parser = argparse.ArgumentParser(description="Deformed petal allignment")
# parser.add_argument("source")
# parser.add_argument("target")
# parser.add_argument("input_image")
# parser.add_argument("deformed_image")
# parser.add_argument("checker_board_image")

# args = parser.parse_args()

# # define the dimensions -> 2D or 3D
# dimensions = 3
# # new thin plate spline object
# thin_plate_spline = itk.ThinPlateSplineKernelTransform[itk.D, Dimension].New()

# # read the source
# source_mesh = itk.meshread(args.source)
# # get the points from the source and convert into float64
# points = itk.array_from_vector_container(source_mesh.GetPoints())
# points = points.astype(np.float64)
# # get transform's source landmarks container
# source = thin_plate_spline.GetSourceLandmarks()
# # set points of the source inside the transforms
# source.SetPoints(itk.vector_container_from_array(points.flatten()))

# # read the target
# target_mesh = itk.meshread(args.target)
# # get the points from the target and convert into float64
# points = itk.array_from_vector_container(target_mesh.GetPoints())
# points = points.astype(np.float64)
# # get transform's target landmarks container
# target = thin_plate_spline.GetTargetLandmarks()
# # set points of the target inside the transforms
# target.SetPoints(itk.vector_container_from_array(points.flatten()))

# # compute the matrix of TPS coefficients needed to define the transform from source to target landmarks.
# thin_plate_spline.ComputeWMatrix()

# # reading the input image
# input_image = itk.imread(args.input_image)

# # read the deformed image
# deformed_image = itk.resample_image_filter(
#     input_image,
#     use_reference_image=True,
#     reference_image=input_image,
#     transform=thin_plate_spline
# )

# # write the deformed_image
# itk.imwrite(deformed_image, args.deformed_image)

# # creates a checkerboard image that juxtaposes the original and deformed images for visual comparison.
# checker_board = itk.checker_board_image_filter(input_image, deformed_image)
# itk.imwrite(checker_board, args.checker_board_image)





