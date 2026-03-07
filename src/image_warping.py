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
import image_utilities as img_util
import JSON_functions as JSONfunc
from os import path, listdir
from skimage import io

def example():
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

def clean_kp(kp):
    kp = np.array(kp).flatten()
    return (int(kp[0]), int(kp[1]))

def display_keypoints(input_dir, output_dir):
    """Process all vein images in the input directory."""
    os.makedirs(output_dir, exist_ok=True)

    image_pairs = img_align.get_file_pairs(input_dir)
    success = 0
    straightened_images = petals_bottom.process_all_petals(input_dir, output_dir, False)
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


        #petal_warp_matrix = [[1,0,int(-petal_x)],[0,1,int(-petal_y)]] # adjust the petal annotation

        #petal_annotation_t = JSONfunc.get_transformed_annotations(petal_annotation, petal_warp_matrix)

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
            #Use the straightened image from petals_bottom
            image = straightened_images[base_name]
            #image = vein_aligned #fallback

            if image is None:
                #print('no image')
                continue
            # Look for corresponding JSON file
            if os.path.exists(json_path):
                #print('getting keypoints')
                keypoints = get_keypoints(vein_annotation_t, vein_aligned)
                #keypoints = get_keypoints(vein_annotation_t, image)
            #print('getting edge keypoints')
            edge_keypoints_top, edge_keypoints_bottom = img_align.get_edge_keypoints(petal_shape, keypoints['top corner'], keypoints['bottom corner'], 
                                                                                     keypoints['center_vein_top'], keypoints['center_vein_bottom'], 200)
            '''
            for keypoint in keypoints:
                #keypoint order is vein_top, vein_bottom, top corner, bottom corner
                print(keypoint)
            '''

            '''
            for keypoint in edge_keypoints_top:
                print('top edge keypoint: ' + str(keypoint))
            for keypoint in edge_keypoints_bottom:
                print('bottom edge keypoint: ' + str(keypoint))
            '''
            
            vis_image = image.copy()
            if len(vis_image.shape) == 2:
                vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2BGR)

            for keypoint in edge_keypoints_top:
                cv2.circle(vis_image, keypoint, 15, (255, 0, 0), -1)  # Blue
                cv2.putText(vis_image, "TOP EDGE", 
                          (keypoint[0] + 20, keypoint[1]), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
            for keypoint in edge_keypoints_bottom:
                cv2.circle(vis_image, keypoint, 15, (0, 255, 0), -1)  # Green
                cv2.putText(vis_image, "BOTTOM EDGE", 
                          (keypoint[0] + 20, keypoint[1]), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2) 

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
                
            

                
            # --- Create a canvas with extra width for the second semi-circle ---
            h, w = vis_image.shape[:2]
            extra_width = w   # or any width you want for the second shape

            canvas = np.zeros((h, w + extra_width, 3), dtype=np.uint8)

            # Copy the original visualization into the left side
            canvas[:, :w] = vis_image

            # --- Draw a second semi-circle on the right side ---
            center_x = (w + extra_width // 2) - 20
            center_y = h // 2
            cv2.circle(canvas, (center_x, center_y), 5, (0, 128, 255), -1)  # Center point
            radius = min(h, extra_width) // 2

            # Draw the semi-circle (180° arc)
            cv2.ellipse(
                canvas,
                (center_x, center_y),
                (radius, radius),
                180,          # rotation
                270, 90,    # startAngle, endAngle
                (0, 128, 255),  # color
                5           # thickness
            )

            # Optional label
            cv2.putText(canvas, "SECOND SEMI-CIRCLE",
                        (center_x - radius, center_y + radius + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 128, 255), 2)
            
            num_circle_keypoints_top = len(edge_keypoints_top)
            num_circle_keypoints_bottom = len(edge_keypoints_bottom)
            circle_keypoints_top, circle_keypoints_bottom = img_util.circle_keypoints(radius, num_circle_keypoints_top, num_circle_keypoints_bottom)
            circle_keypoints_top = img_util.rotate_points_numpy(circle_keypoints_top, degrees=270)
            circle_keypoints_bottom = img_util.rotate_points_numpy(circle_keypoints_bottom, degrees=270)
            offset = w
            for keypoint in circle_keypoints_top:
                x, y = keypoint

                # translate from local circle coordinates to arc center
                x_new = int(center_x + x)
                y_new = int(center_y + y)

                cv2.circle(canvas, (x_new, y_new), 15, (0, 0, 255), -1)
                cv2.putText(canvas, "TOP EDGE",
                            (x_new + 20, y_new),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

            for keypoint in circle_keypoints_bottom:
                x, y = keypoint

                x_new = int(center_x + x)
                y_new = int(center_y + y)

                cv2.circle(canvas, (x_new, y_new), 15, (0, 255, 0), -1)
                cv2.putText(canvas, "BOTTOM EDGE",
                            (x_new + 20, y_new),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2) 

            # Display the combined image
            io.imshow(canvas)
            io.show()

            #io.imshow(vis_image)
            #io.show()
        except Exception as e:
            import traceback
            traceback.print_exc()

def warp_petals(input_dir, output_dir):
    """Process all vein images in the input directory."""
    os.makedirs(output_dir, exist_ok=True)

    show = input("Show overlaid images? (y/n): ")

    image_pairs = img_align.get_file_pairs(input_dir)
    success = 0
    straightened_images = petals_bottom.process_all_petals(input_dir, output_dir, False)
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


        #petal_warp_matrix = [[1,0,int(-petal_x)],[0,1,int(-petal_y)]] # adjust the petal annotation

        #petal_annotation_t = JSONfunc.get_transformed_annotations(petal_annotation, petal_warp_matrix)

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
            #Use the straightened image from petals_bottom
            image = straightened_images[base_name]
            #image = vein_aligned #fallback

            if image is None:
                #print('no image')
                continue
            # Look for corresponding JSON file
            if os.path.exists(json_path):
                #print('getting keypoints')
                keypoints = get_keypoints(vein_annotation_t, vein_aligned)
                #keypoints = get_keypoints(vein_annotation_t, image)


            edge_keypoints_top, edge_keypoints_bottom = img_align.get_edge_keypoints(petal_shape, keypoints['top corner'], keypoints['bottom corner'], 
                                                                                     keypoints['center_vein_top'], keypoints['center_vein_bottom'], 100)
            
            # If source top runs opposite to dest top arc, reverse it
            edge_keypoints_top = edge_keypoints_top[::-1]

            # Similarly for bottom if needed
            edge_keypoints_bottom = edge_keypoints_bottom[::-1]
            
            '''
            for keypoint in keypoints:
                #keypoint order is vein_top, vein_bottom, top corner, bottom corner
                print(keypoint)
                #keypoint order: vein_top, vein_bottom, top_corner, bottom_corner
            '''
            '''
            for keypoint in edge_keypoints_top:
                print('top edge keypoint: ' + str(keypoint))
            for keypoint in edge_keypoints_bottom:
                print('bottom edge keypoint: ' + str(keypoint))
            '''
            #print(keypoints.values())
            #add keypoints to np.array
            src_points = np.array([keypoints['center_vein_top'], keypoints['center_vein_bottom'], keypoints['top corner'], keypoints['bottom corner']])
            src_points = np.concatenate((src_points, edge_keypoints_top, edge_keypoints_bottom), axis=0)
            #warp to semi circle
            radius = 400
            num_circle_keypoints_top = len(edge_keypoints_top)
            num_circle_keypoints_bottom = len(edge_keypoints_bottom)
            circle_keypoints_top, circle_keypoints_bottom = img_util.circle_keypoints(radius, num_circle_keypoints_top, num_circle_keypoints_bottom)

            # Define where the semi-circle center should sit in the output image
            h, w = image.shape[:2]
            cx, cy = w // 2, h // 2  # place the arc center in the middle of the output

            # Now build dest_points offset by (cx, cy)
            dest_points = np.array([
                [cx + radius, cy],   # center_vein_top  -> arc tip
                [cx,          cy],   # center_vein_bottom -> arc center (flat edge)
                [cx,  cy - radius],  # top corner        -> left end of diameter
                [cx,  cy + radius],  # bottom corner     -> right end of diameter
            ])

            # Also offset the circle edge keypoints
            circle_keypoints_top_abs = circle_keypoints_top    + np.array([cx, cy])
            circle_keypoints_bottom_abs = circle_keypoints_bottom + np.array([cx, cy])
            circle_keypoints_top_abs = img_util.rotate_points_numpy(circle_keypoints_top_abs, origin=(cx, cy), degrees=270)
            circle_keypoints_bottom_abs = img_util.rotate_points_numpy(circle_keypoints_bottom_abs, origin=(cx, cy), degrees=270)

            dest_points = np.concatenate( (dest_points, circle_keypoints_top_abs, circle_keypoints_bottom_abs), axis=0 )

            #src_tps  = src_points[:,  ::-1]   # swap to (row, col)
            #dest_tps = dest_points[:, ::-1]   # swap to (row, col)

            #dest_points = np.array([[radius,0], [0,0], [0,-radius], [0, radius]])
            #dest_points = np.concatenate((dest_points, circle_keypoints_top, circle_keypoints_bottom), axis=0)
            print('beginning TPS deformation on: ' + base_name)
            tps = ThinPlateSplineTransform()
            success = tps.estimate(dest_points, src_points)  # Note inverse mapping for warp()
            #success = tps.estimate(dest_tps, src_tps)  # Note inverse mapping for warp()
            if not success:
                print(f"TPS estimation failed for {base_name}")
                continue

            # Warp the image using the TPS transform
            warped = warp(image, tps)
            #io.imshow(warped)
            #io.show()
            edge_keypoints_top = np.array(edge_keypoints_top)
            edge_keypoints_bottom = np.array(edge_keypoints_bottom)
            
            # Display original and warped images with landmarks
            print('displaying results for: ' + base_name)
            fig, (ax1, ax2) = plt.subplots(1, 2)
            ax1.imshow(image, cmap='gray')
            #ax1.scatter(src_points[:, 0], src_points[:, 1], marker='x', color='red')
            ax1.scatter(edge_keypoints_top[:, 0], edge_keypoints_top[:, 1], marker='o', color='blue', label='Top Edge Keypoints')
            ax1.scatter(edge_keypoints_bottom[:, 0], edge_keypoints_bottom[:, 1], marker='o', color='green', label='Bottom Edge Keypoints')
            #for i, point in enumerate(src_points):
            #    ax1.annotate(f'KP{i+1}', (point[0] + 5, point[1] - 5), color='red', fontsize=12)
            ax1.set_title('Original Image')

            ax2.imshow(warped, cmap='gray')
            #ax2.scatter(dest_points[:, 0], dest_points[:, 1], marker='x', color='red')
            ax2.scatter(circle_keypoints_top_abs[:, 0], circle_keypoints_top_abs[:, 1], marker='o', color='blue', label='Top Edge Keypoints')
            ax2.scatter(circle_keypoints_bottom_abs[:, 0], circle_keypoints_bottom_abs[:, 1], marker='o', color='green', label='Bottom Edge Keypoints')
            #for i, point in enumerate(dest_points):
            #    ax2.annotate(f'KP{i+1}', (point[0] + 5, point[1] - 5), color='red', fontsize=12)
            ax2.set_title('Warped Image')

            for i, point in enumerate(edge_keypoints_top):
                ax1.annotate(str(i), (point[0]+5, point[1]-5), color='blue', fontsize=8)
            for i, point in enumerate(circle_keypoints_top_abs):
                ax2.annotate(str(i), (point[0]+5, point[1]-5), color='blue', fontsize=8)

            for i, point in enumerate(edge_keypoints_bottom):
                ax1.annotate(str(i), (point[0]+5, point[1]-5), color='green', fontsize=8)
            for i, point in enumerate(circle_keypoints_bottom_abs):
                ax2.annotate(str(i), (point[0]+5, point[1]-5), color='green', fontsize=8)
            if show.lower() == 'y':
                plt.show()
            
            output_path = os.path.join(output_dir, f"{base_name}_warped.png")
            plt.savefig(output_path, bbox_inches='tight', dpi=150)
            plt.close()  # Free memory, important when processing many images


        except Exception as e:
            import traceback
            traceback.print_exc()
            

if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(1)
    #example cmd line: python3 .\image_warping.py ..\example_dataset ..\output
    #example()
    #display_keypoints(sys.argv[1], sys.argv[2])
    warp_petals(sys.argv[1], sys.argv[2])