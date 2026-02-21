import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage.transform import ThinPlateSplineTransform, warp
import petals_bottom
import os
import sys
import cv2
import glob
import json
import image_alignment as img_align

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

def get_keypoints(path, image):
    keypoints = petals_bottom.load_keypoints(path)
    petal_mask = petals_bottom.get_petal_shape_simple(image)
    edge_points, edge_type, bounds = petals_bottom.detect_edge_from_keypoints(petal_mask, keypoints)
    straightened, edge_pts, etype, perp_pt1, perp_pt2, vec_dir = petals_bottom.straighten_edge(image, petal_mask, edge_points, edge_type, bounds, keypoints)
    corners = petals_bottom.get_corners(perp_pt1, perp_pt2, image)
    corner_keypoints = {'top corner': corners[0], 'bottom corner': corners[1]}
    #Given that we cut the petal at the bottom the center_vein_bottom point is no loner on the petal
    bottom_pt = np.array(keypoints['center_vein_bottom'], dtype=np.float32)
    top_pt = np.array(keypoints['center_vein_top'], dtype=np.float32)
    #compute intersection point of vein axis and perp line. This is where the new bottom point will be
    new_bottom = petals_bottom.intersection(bottom_pt, top_pt, perp_pt1, perp_pt2)
    keypoints['center_vein_bottom'] = new_bottom
    keypoints = keypoints | corner_keypoints
    return keypoints

def warp(keypoints):
    src = np.array(keypoints)
    pass

def warp_petals(input_dir, output_dir):
    """Process all vein images in the input directory."""
    os.makedirs(output_dir, exist_ok=True)
    
    patterns = ["*.JPG", "*.jpg", "*.png", "*.PNG"]
    image_files = []
    for pattern in patterns:
        image_files.extend(glob.glob(os.path.join(input_dir, pattern)))
    
    # Filter for vein images
    vein_files = [f for f in image_files if 'Vein' in os.path.basename(f) or 'vein' in os.path.basename(f)]
    
    if len(vein_files) == 0:
        return
    for i, img_path in enumerate(vein_files):
        filename = os.path.basename(img_path)
        
        try:
            # Read image
            #print("reading image")
            image = cv2.imread(img_path)
            if image is None:
                #print('no image')
                continue
            
            # Look for corresponding JSON file
            base_name = os.path.splitext(filename)[0]
            json_path = os.path.join(input_dir, f"{base_name}_labels.json")
            
            keypoints = None
            if os.path.exists(json_path):
                #print('getting keypoints')
                keypoints = get_keypoints(json_path, image)
            for keypoint in keypoints:
                #keypoint order is vein_top, vein_bottom, top corner, bottom corner
                #print(keypoint)
                pass
        except:
            pass

if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(1)
    #example cmd line: python3 .\image_warping.py ..\example_dataset ..\output
    warp_petals(sys.argv[1], sys.argv[2])