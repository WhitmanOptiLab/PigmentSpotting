"""
Straighten the outer edge of petal/vein images to create a perfectly straight line.
Uses JSON keypoints to identify the correct bottom edge.
"""

import os
import sys
import cv2
from matplotlib import image
import numpy as np
import glob
import json


def load_keypoints(json_path):
    """Load center_vein_bottom and center_vein_top from JSON labels."""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Get the first (and only) entry
        entry = list(data.values())[0]
        regions = entry['regions']
        
        keypoints = {}
        for region in regions.values():
            label = region['region_attributes'].get('label', region['region_attributes'].get('label ', '')).strip()
            cx = region['shape_attributes']['cx']
            cy = region['shape_attributes']['cy']
            keypoints[label] = (cx, cy)
        
        return keypoints
    except Exception as e:
        return None


def get_petal_shape_simple(image):
    """Simple petal shape detection."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    
    label_count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 4, cv2.CV_32S)
    
    if label_count < 2:
        return binary
    
    sizes = stats[:, -1]
    max_label = np.argmax(sizes[1:]) + 1
    petal_mask = (labels == max_label).astype(np.uint8) * 255
    
    return petal_mask


def detect_edge_from_keypoints(petal_mask, keypoints):
    """
    Detect which edge to straighten based on center_vein_bottom keypoint.
    """
    contours, _ = cv2.findContours(petal_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if not contours:
        raise ValueError("No contours found")
    
    largest_contour = max(contours, key=cv2.contourArea)
    contour_points = largest_contour.reshape(-1, 2)
    
    x_min, y_min = np.min(contour_points, axis=0)
    x_max, y_max = np.max(contour_points, axis=0)
    width = x_max - x_min
    height = y_max - y_min
    
    # Use keypoints to determine which edge is the "bottom"
    if keypoints and 'center_vein_bottom' in keypoints and 'center_vein_top' in keypoints:
        bottom_pt = keypoints['center_vein_bottom']
        top_pt = keypoints['center_vein_top']
        
        # Determine orientation based on keypoints
        # If bottom is to the left of top -> left edge is bottom
        # If bottom is to the right of top -> right edge is bottom
        # If bottom is below top -> bottom edge is bottom
        # If bottom is above top -> top edge is bottom
        
        dx = bottom_pt[0] - top_pt[0]
        dy = bottom_pt[1] - top_pt[1]
        
        if abs(dx) > abs(dy):
            # Horizontal orientation
            if dx < 0:
                # Bottom is to the left
                edge_type = 'left'
                threshold = x_min + 0.35 * width
                edge_points = contour_points[contour_points[:, 0] <= threshold]
            else:
                # Bottom is to the right
                edge_type = 'right'
                threshold = x_max - 0.35 * width
                edge_points = contour_points[contour_points[:, 0] >= threshold]
        else:
            # Vertical orientation
            if dy > 0:
                # Bottom is below (higher y value)
                edge_type = 'bottom'
                threshold = y_max - 0.35 * height
                edge_points = contour_points[contour_points[:, 1] >= threshold]
            else:
                # Bottom is above (lower y value)
                edge_type = 'top'
                threshold = y_min + 0.35 * height
                edge_points = contour_points[contour_points[:, 1] <= threshold]
        
    else:
        # Fallback: prefer left edge for vein images
        edge_type = 'left'
        threshold = x_min + 0.35 * width
        edge_points = contour_points[contour_points[:, 0] <= threshold]
    
    return edge_points, edge_type, (x_min, y_min, x_max, y_max)


def straighten_edge(image, petal_mask, edge_points, edge_type, bounds, keypoints):
    """
    Create a perpendicular line that passes through the bottom edge of the petal on both sides.
    """
    h, w = image.shape[:2]
    
    if not keypoints or 'center_vein_bottom' not in keypoints or 'center_vein_top' not in keypoints:
        return image, edge_points, edge_type, None, None, None
    
    bottom_pt = np.array(keypoints['center_vein_bottom'], dtype=np.float32)
    top_pt = np.array(keypoints['center_vein_top'], dtype=np.float32)
    
    # Vector from top to bottom
    vec = bottom_pt - top_pt
    vec_length = np.linalg.norm(vec)
    
    if vec_length == 0:
        return image, edge_points, edge_type, None, None, None
    
    # Normalize the vector
    vec_normalized = vec / vec_length
    
    # Perpendicular vector (rotate 90 degrees)
    perp_vec = np.array([-vec_normalized[1], vec_normalized[0]])
    
    # Find the best position for the perpendicular line that passes through bottom edge points
    # Project all edge points onto the TOP-BOTTOM direction
    projections = np.array([np.dot(pt - top_pt, vec_normalized) for pt in edge_points])
    
    # Get bottom 98% of edge points (larger percentage to move line further up onto petal)
    max_proj = np.max(projections)
    min_proj = np.min(projections)
    proj_range = max_proj - min_proj
    threshold = max_proj - 0.98 * proj_range
    bottom_mask = projections >= threshold
    bottom_edge_points = edge_points[bottom_mask]
    
    # For each bottom edge point, find its projection onto the yellow line
    # Then take the mean of these projections
    bottom_projections = projections[bottom_mask]
    mean_bottom_projection = np.mean(bottom_projections)
    
    # The perpendicular line passes through this point on the yellow line
    cut_line_point = top_pt + mean_bottom_projection * vec_normalized
    
    # Create the perpendicular line endpoints
    line_length = max(w, h) * 2
    perp_pt1 = cut_line_point - perp_vec * line_length
    perp_pt2 = cut_line_point + perp_vec * line_length
    
    # Create a mask - keep only pixels on the TOP side of the perpendicular line
    result = image.copy()
    
    # Check which side top_pt is on
    top_side = np.dot(vec_normalized, top_pt - cut_line_point)
    # Create mask by checking each pixel
    for y in range(h):
        for x in range(w):
            point = np.array([x, y], dtype=np.float32)
            side = np.dot(vec_normalized, point - cut_line_point)
            
            # Keep pixels on the same side as top_pt
            if np.sign(side) != np.sign(top_side):
                result[y, x] = 0
            '''
            if (top_side > 0 and side < 0) or (top_side < 0 and side > 0):
                result[y, x] = 0  # Set to black
            '''
    
    return result, edge_points, edge_type, perp_pt1, perp_pt2, vec_normalized


def process_all_petals(input_dir, output_dir):
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
    
    success = 0
    for i, img_path in enumerate(vein_files):
        filename = os.path.basename(img_path)
        
        try:
            # Read image
            image = cv2.imread(img_path)
            if image is None:
                continue
            
            # Look for corresponding JSON file
            base_name = os.path.splitext(filename)[0]
            json_path = os.path.join(input_dir, f"{base_name}_labels.json")
            
            keypoints = None
            if os.path.exists(json_path):
                keypoints = load_keypoints(json_path)
            
            # Process image
            petal_mask = get_petal_shape_simple(image)
            edge_points, edge_type, bounds = detect_edge_from_keypoints(petal_mask, keypoints)
            straightened, edge_pts, etype, perp_pt1, perp_pt2, vec_dir = straighten_edge(image, petal_mask, edge_points, edge_type, bounds, keypoints)

            line_pt, line_dir, line_normal = fit_bottom_edge_line(edge_points)

            
            # Create visualization
            vis_image = image.copy()
            
            h, w = image.shape[:2]
            L = max(h, w) * 2

            line_p1 = (line_pt - line_dir * L).astype(int)
            line_p2 = (line_pt + line_dir * L).astype(int)

            cv2.line(vis_image, tuple(line_p1),tuple(line_p2),(255, 0, 0), 4)

            normal_len = 150
            normal_end = (line_pt + line_normal * normal_len).astype(int)

            cv2.arrowedLine(vis_image, tuple(line_pt.astype(int)), tuple(normal_end), (0, 165, 255), 4, tipLength=0.2)


            # Draw keypoints if available
            if keypoints:
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
                
                # Draw line from top to bottom
                if 'center_vein_top' in keypoints and 'center_vein_bottom' in keypoints:
                    cv2.line(vis_image, keypoints['center_vein_top'], keypoints['center_vein_bottom'], 
                            (0, 255, 255), 3)  # Yellow line
            
            # Draw perpendicular cutting line if available
            if perp_pt1 is not None and perp_pt2 is not None:
                pt1 = tuple(perp_pt1.astype(int))
                pt2 = tuple(perp_pt2.astype(int))
                cv2.line(vis_image, pt1, pt2, (0, 0, 255), 4)  # Red perpendicular line
            
            # Draw edge points
            for pt in edge_pts[::max(1, len(edge_pts)//40)]:
                cv2.circle(vis_image, tuple(pt.astype(int)), 5, (0, 255, 0), -1)
            
            # Save
            ext = os.path.splitext(filename)[1]
            output_path = os.path.join(output_dir, f"{base_name}_Straightened{ext}")
            viz_path = os.path.join(output_dir, f"{base_name}_EdgeDetection{ext}")
            
            cv2.imwrite(output_path, straightened)
            cv2.imwrite(viz_path, vis_image)
            
            success += 1
            
        except Exception as e:
            import traceback
            traceback.print_exc()

def fit_bottom_edge_line(edge_points, trim_percent=0.15):
    """
    Fit a straight line to the bottom edge using total least squares (PCA).

    Returns:
        line_point: (x, y) point on the fitted line
        line_dir: normalized direction vector of the line
        normal: normalized normal vector (points "up" the petal)
    """
    if len(edge_points) < 10:
        raise ValueError("Not enough edge points to fit a line")

    pts = edge_points.astype(np.float32)

    # --- Optional trimming to remove extreme outliers ---
    # Sort by projection along principal axis later; first rough center
    center = np.mean(pts, axis=0)
    dists = np.linalg.norm(pts - center, axis=1)

    lo = np.percentile(dists, trim_percent * 100)
    hi = np.percentile(dists, (1 - trim_percent) * 100)
    pts = pts[(dists >= lo) & (dists <= hi)]

    # --- PCA / total least squares ---
    mean = np.mean(pts, axis=0)
    pts_centered = pts - mean

    _, _, vt = np.linalg.svd(pts_centered)
    direction = vt[0]          # principal axis
    direction /= np.linalg.norm(direction)

    # Normal to the line
    normal = np.array([-direction[1], direction[0]])
    normal /= np.linalg.norm(normal)

    return mean, direction, normal

def cut_image_with_line(image, line_point, normal, keep_point):
    """
    Cuts the image along a line.
    keep_point determines which side to keep (e.g. center_vein_top).
    """
    h, w = image.shape[:2]
    result = image.copy()

    keep_side = np.dot(normal, keep_point - line_point)

    for y in range(h):
        for x in range(w):
            p = np.array([x, y], dtype=np.float32)
            side = np.dot(normal, p - line_point)

            if np.sign(side) != np.sign(keep_side):
                result[y, x] = 0

    return result

def process_petals_linear(input_dir, output_dir):
    # For each image, detect edge points, fit line, then cut
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
    
    success = 0
    for i, img_path in enumerate(vein_files):
        filename = os.path.basename(img_path)
        
        try:
            # Read image
            image = cv2.imread(img_path)
            if image is None:
                continue
            
            # Look for corresponding JSON file
            base_name = os.path.splitext(filename)[0]
            json_path = os.path.join(input_dir, f"{base_name}_labels.json")
            
            keypoints = None
            if os.path.exists(json_path):
                keypoints = load_keypoints(json_path)
            
            # Process image
            petal_mask = get_petal_shape_simple(image)
            edge_points, edge_type, bounds = detect_edge_from_keypoints(petal_mask, keypoints)
            line_pt, line_dir, normal = fit_bottom_edge_line(edge_points)
            
            # Create visualization
            vis_image = image.copy()

            if keypoints:
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
                
                # Draw line from top to bottom
                if 'center_vein_top' in keypoints and 'center_vein_bottom' in keypoints:
                    cv2.line(vis_image, keypoints['center_vein_top'], keypoints['center_vein_bottom'], 
                            (0, 255, 255), 3)  # Yellow line
                    
            h, w = image.shape[:2]
            L = max(h, w) * 2

            p1 = (line_pt - line_dir * L).astype(int)
            p2 = (line_pt + line_dir * L).astype(int)

            cv2.line(vis_image, tuple(p1), tuple(p2), (255, 0, 0), 3)
            keep_pt = np.array(keypoints['center_vein_top'], dtype=np.float32)
            straightened = cut_image_with_line(image, line_pt, normal, keep_pt)

        except Exception as e:
            import traceback
            traceback.print_exc()


    pass


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(1)
    #example cmd line python3 .\petals_bottom.py ..\example_dataset ..\output
    process_all_petals(sys.argv[1], sys.argv[2])

