import cv2
import numpy as np

def resize_image(img, size):
    """Resize an image.

    Args:
    img: The image to be resized.
    size: The target size.

    Returns:
        The resized image.
    """
    r = size / img.shape[1]
    dim = (size, int(img.shape[0] * r))
    return cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

def match_image_size(img1, img2):
    """Resize an image.

    Args:
    img1: The image with a target size.
    img2: The image to be resized.

    Returns:
        img2 resized to fit img1.
    """
    dim = (img1.shape[1])
    return cv2.resize(img2, dim, interpolation = cv2.INTER_AREA)

def brighten_image(img, alpha, beta):
    """Brighten an image.

    Args:
    img: The image to be brightened.
    alpha: The factor to brighten the image by.
    beta: A constant value to add to the brightened image.

    Returns:
        The brightened image.
    """
    new_image = (np.clip(alpha*(img.astype(np.int32)) + beta, 0, 255)).astype(img.dtype)
    return new_image

def make_bw(img):
    """Convert an image to black and white.

    Args:
    img: The image to be converted to black and white.

    Returns:
        The black and white image.
    """
    new_image = np.zeros(img.shape, img.dtype)
    new_image = np.clip(~img, 0, 1)*255
    return new_image

def remove_edge(img):
    """
    This function removes an edge present in many of the images, which causes shape detection to go awry if
    the petal touches the edge.

    Args:
    img: the full vein image in color

    Returns: image with the left edge cut out
    """
    dst = cv2.Canny(img, 50, 200, None, 3) # detect edges in image
    linesP = cv2.HoughLinesP(dst, 1, np.pi / 180, 100, 20, 100) # detect lines within image
    longest_line = None

    if linesP is not None:
        longest_line = max(linesP, key = \
                           lambda line : ((line[0][2] - line[0][0])**2 + (line[0][3] - line[0][1])**2)**(1/2))

    if longest_line is not None:
        line_angle = np.degrees(np.arctan2((longest_line[0][1] - longest_line[0][3]) , (longest_line[0][0] - longest_line[0][2])))
        
        if line_angle == 90.0:
            center_vertical = img.shape[1]//2

            if longest_line[0][0] < center_vertical or longest_line[0][2] < center_vertical: # on left side
                pts = np.array([
                        (0,0),
                        (0,img.shape[0]),
                        (longest_line[0][2], img.shape[0]),
                        (longest_line[0][0], 0)
                    ], np.int32)
            else: # if on right side
                pts = np.array([
                        (longest_line[0][2], img.shape[0]),
                        (longest_line[0][0], 0),
                        (img.shape[1], 0),
                        (img.shape[1], img.shape[0])
                    ], np.int32)  
                
                # Fill the polygon
            cv2.fillPoly(img, [pts], 0)
    
    return img

def is_point_below_line(point_p, point1, point2):
    """
    Checks if point_p is below the line defined by point1 and point2.
    Assumes a standard image coordinate system (y increases downwards).
    """
    xp, yp = point_p
    x1, y1 = point1
    x2, y2 = point2

    # Handle vertical lines to prevent division by zero
    if x2 == x1:
        # If vertical, "below" can be defined by the x position relative to a reference (e.g., left/right)
        # Here we consider 'below' to be relative to the y-axis, which is not applicable.
        # This function focuses on non-vertical lines for 'below'/'above' comparison.
        return None 

    # Calculate slope (a) and y-intercept (b)
    a = (y2 - y1) / (x2 - x1)
    b = y1 - a * x1

    # Calculate the y-value on the line at the point's x-coordinate
    y_on_line = a * xp + b

    # In a typical image coordinate system, y increases as you go down.
    # So, yp > y_on_line means the point is "below" the line.
    # yp < y_on_line means the point is "above" the line.
    if yp > y_on_line:
        return True  # Point is below the line
    elif yp < y_on_line:
        return False # Point is above the line
    else:
        return None  # Point is on the line
    
def is_left_of_line(point1, point2, point_p):
    """
    Check if point P is to the left of the directed line from A to B using the cross product.
    A, B, P are tuples/lists/objects with x and y attributes (e.g., (x, y)).
    """
    # Vector AB: (Bx - Ax, By - Ay)
    # Vector AP: (Px - Ax, Py - Ay)
    # Cross product magnitude: (Bx - Ax) * (Py - Ay) - (By - Ay) * (Px - Ax)
    cross_product = (point2[0] - point1[0]) * (point_p[1] - point1[1]) - (point2[1] - point1[1]) * (point_p[0] - point1[0])
    
    # Use a small threshold (epsilon) for floating-point comparisons
    threshold = 1e-9
    if cross_product > threshold:
        return True  # Left
    elif cross_product < -threshold:
        return False # Right
    else:
        # Point is on the line
        # Depending on exact requirements, you might want to return "on_line" or treat it as True/False
        return False # Or True, as per specific logic needs
    
def circle_keypoints(radius, n_points_top, n_points_bottom):
    # 1. Generate N angles from 0 to pi (180 degrees)
    angles_top = np.linspace(0, np.pi/2, n_points_top + 2)[1:-1]  # Exclude the endpoints
    angles_bottom = np.linspace(np.pi/2, np.pi, n_points_bottom + 2)[1:-1]  # Exclude the endpoints
    
    # 2. Calculate x and y coordinates
    keypoints_top = []  
    for angle in angles_top:
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        keypoints_top.append((x, y))
    keypoints_bottom = []
    for angle in angles_bottom:
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        keypoints_bottom.append((x, y))
    return keypoints_top, keypoints_bottom

def rotate_points_numpy(points, origin=(0, 0), degrees=0):
    """Rotates an array of points counterclockwise around an origin."""
    angle = np.deg2rad(degrees)
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle), np.cos(angle)]])
    o = np.atleast_2d(origin)
    p = np.atleast_2d(points)
    return np.squeeze((R @ (p.T - o.T) + o.T).T)

def angle_from_center(pt, center):
    return np.arctan2(pt[1] - center[1], pt[0] - center[0])

    
    
