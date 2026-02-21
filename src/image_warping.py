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
import JSON_functions as JSONfunc
from os import path, listdir
from skimage import io

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

def get_keypoints(annotation, image):
    keypoints = petals_bottom.get_keypoints(annotation)
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

    image_pairs = img_align.get_file_pairs(input_dir)
    success = 0
    for pair in image_pairs:
        #print(f"Processing pair: {pair[0]} and {pair[1]}")
        if "vein" in pair[0].lower():
            vein_img_filename = pair[0]
            petal_img_filename = pair[1]
        else:
            vein_img_filename = pair[1]
            petal_img_filename = pair[0]

        petal_image, petal_annotation = JSONfunc.img_crop(petal_img_filename, input_dir)
        
        petal_x = petal_annotation["bounding_box"]["x"]
        petal_y = petal_annotation["bounding_box"]["y"]


        petal_warp_matrix = [[1,0,int(-petal_x)],[0,1,int(-petal_y)]] # adjust the petal annotation

        petal_annotation_t = JSONfunc.get_transformed_annotations(petal_annotation, petal_warp_matrix)

        #vein initalization for image (vein_image) and dictionary (new_vein_dict)

        vein_annotation = JSONfunc.parse_annotation(vein_img_filename, input_dir, group_attr="label")
        vein_image = cv2.imread(path.join(input_dir, vein_img_filename),0)

        img_path = os.path.join(input_dir, vein_img_filename)
        filename = vein_img_filename

        
        try:
            # Read image
            image = vein_image
            #image = cv2.imread(img_path)
            if image is None:
                continue
            #image is vein image
            # Look for corresponding JSON file
            base_name = os.path.splitext(filename)[0]
            json_path = os.path.join(input_dir, f"{base_name}_labels.json")
            
            keypoints = None
            if os.path.exists(json_path):
                keypoints = petals_bottom.load_keypoints(json_path)
            
            # Process image
            petal_shape, vein_aligned, warp_matrix = img_align.align_images(petal_image, image)
            inv_warp_matrix = cv2.invertAffineTransform(warp_matrix) 
            vein_annotation_t = JSONfunc.get_transformed_annotations(vein_annotation,inv_warp_matrix)
            aligned_keypoints = petals_bottom.get_keypoints(vein_annotation_t)
            keypoints = aligned_keypoints
            #keypoints = vein_annotation_t
            image = vein_aligned
            if image is None:
                #print('no image')
                continue
            # Look for corresponding JSON file
            if os.path.exists(json_path):
                #print('getting keypoints')
                keypoints = get_keypoints(vein_annotation_t, image)
            for keypoint in keypoints:
                #keypoint order is vein_top, vein_bottom, top corner, bottom corner
                print(keypoint)
                vis_image = image.copy()
                if 'center_vein_bottom' in keypoints:
                    cv2.circle(vis_image, keypoints['center_vein_bottom'], 15, (255, 0, 255), -1)  # Magenta
                    cv2.putText(vis_image, "BOTTOM", 
                              (keypoints['center_vein_bottom'][0] + 20, keypoints['center_vein_bottom'][1]), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 2)
                if 'center_vein_top' in keypoints:
                    cv2.circle(vis_image, keypoints['center_vein_top'], 15, (255, 255, 0), -1)  # Cyan
                    cv2.putText(vis_image, "TOP", 
                              (keypoints['center_vein_top'][0] + 20, keypoints['center_vein_top'][1]), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
                if 'top corner' in keypoints:
                    cv2.circle(vis_image, keypoints['top corner'], 15, (255, 255, 0), -1)  # Cyan
                    cv2.putText(vis_image, "TOP Corner", 
                              (keypoints['top corner'][0] + 20, keypoints['top corner'][1]), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
                if 'bottom corner' in keypoints:
                    cv2.circle(vis_image, keypoints['bottom corner'], 15, (255, 255, 0), -1)  # Cyan
                    cv2.putText(vis_image, "BOTTOM Corner", 
                              (keypoints['bottom corner'][0] + 20, keypoints['bottom corner'][1]), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
                # Draw line from top to bottom
                if 'center_vein_top' in keypoints and 'center_vein_bottom' in keypoints:
                    cv2.line(vis_image, keypoints['center_vein_top'], keypoints['center_vein_bottom'], 
                            (0, 255, 255), 3)  # Yellow line
                if 'top corner' in keypoints and 'bottom corner' in keypoints:
                    cv2.line(vis_image, keypoints['top corner'], keypoints['bottom corner'], 
                            (0, 255, 255), 3)  # Yellow line
                io.imshow(vis_image)
                io.show()
        except:
            pass

if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(1)
    #example cmd line: python3 .\image_warping.py ..\example_dataset ..\output
    warp_petals(sys.argv[1], sys.argv[2])