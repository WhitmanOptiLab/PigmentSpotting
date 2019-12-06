import cv2
import sys
import numpy as np
from skimage import io

def resize_image(img, s):
    r = s / img.shape[1]
    dim = (s, int(img.shape[0] * r))
    return cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

def match_image_size(im1, im2):
    dim = (im1.shape[1])
    return cv2.resize(im2, dim, interpolation = cv2.INTER_AREA)

def get_image_kmeans(image, k):
    Z = image.reshape((-1,3))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = int(k)
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((image.shape))
    return res2

def brighten_image(i, alpha, beta):
    new_image = np.zeros(i.shape, i.dtype)
    for y in range(i.shape[0]):
        for x in range(i.shape[1]):
            for c in range(i.shape[2]):
                new_image[y,x,c] = np.clip(alpha*i[y,x,c] + beta, 0, 255)
    return new_image

def make_bw(img):
    new_image = np.zeros(img.shape, img.dtype)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if img[y,x] == 255:
                new_image[y,x] =0
            else:
                new_image[y,x] = 255
    return new_image

def process_vein(im1, im2):
    grimg = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
    background = np.zeros(grimg.shape, grimg.dtype)
    x_offset=y_offset=100
    background[y_offset:y_offset+im2.shape[0], x_offset:x_offset+im2.shape[1]] = im2
    return background

def get_petal_shape(im):
    b_im = brighten_image(im, 2,0)

    kmeans = get_image_kmeans(b_im, 3)

    im_in =  cv2.cvtColor(kmeans,cv2.COLOR_BGR2GRAY)
    im_in = make_bw(im_in)

    im_in = cv2.bitwise_not(im_in)

    im_blurred = cv2.blur(im_in, (60,60))

    kernel = np.ones((2,2), np.uint8)

    im_closed = cv2.dilate(im_in, kernel, iterations=1)
    im_closed = cv2.erode(im_closed, kernel, iterations=1)

    im_closed = cv2.blur(im_closed, (20,20))

    # Threshold.
    # Set values equal to or above 220 to 0.
    # Set values below 220 to 255.

    th, im_th = cv2.threshold(im_closed, 220, 255, cv2.THRESH_BINARY_INV)
    th, im_th2 = cv2.threshold(im_th, 220, 255, cv2.THRESH_BINARY_INV)
    # Copy the thresholded image.
    im_floodfill = im_th.copy()
    im_floodfill1 = im_th2.copy()
    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = im_th.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)

    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0,0), 255);
    cv2.floodFill(im_floodfill1, mask, (0,0), 255);
    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    im_out = im_th | im_floodfill_inv

    return im_out

def get_vein_shape(im):

    im_in = cv2.bitwise_not(im)

    im_blurred = cv2.blur(im_in, (60,60))
    im_blurred = cv2.blur(im_blurred, (60,60))
    # im_blurred = cv2.blur(im_blurred, (60,60))

    kernel = np.ones((2,2), np.uint8)

    im_closed = cv2.dilate(im_blurred, kernel, iterations=1)
    im_closed = cv2.erode(im_closed, kernel, iterations=1)

    im_closed = cv2.blur(im_closed, (20,20))
    # Threshold.
    # Set values equal to or above 220 to 0.
    # Set values below 220 to 255.

    th, im_th = cv2.threshold(im_closed, 220, 255, cv2.THRESH_BINARY_INV)

    # Copy the thresholded image.
    im_floodfill = im_th.copy()

    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = im_th.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)

    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0,0), 255);

    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    # Combine the two images to get the foreground.
    im_out = im_th | im_floodfill_inv
    return im_th

def align(im1,im2,s1,s2):
    sz = im1.shape
    warp_mode = cv2.MOTION_EUCLIDEAN

    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else:
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    number_of_iterations = 30000

    termination_eps = 1e-5

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)

    (cc, warp_matrix) = cv2.findTransformECC(s1,s2,warp_matrix, warp_mode, criteria, None, 1)

    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        im2_aligned = cv2.warpPerspective(im2, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else:
        im2_aligned = cv2.warpAffine(im2, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

    return im2_aligned

def align2(im1,im2,s1,s2):
    MAX_FEATURES = 200
    GOOD_MATCH_PERCENT = 0.15

    im1Gray = s1
    im2Gray = s2

    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    matches.sort(key=lambda x: x.distance, reverse=False)

    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
    cv2.imwrite("matches.jpg", imMatches)

    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    height, width= im2.shape
    im1Reg = cv2.warpPerspective(im1, h, (width, height))

    io.imshow(im1Reg)
    io.show()

def main():
    img1 = cv2.imread(sys.argv[1])
    img2 = cv2.imread(sys.argv[2], cv2.IMREAD_GRAYSCALE)
    img1 = resize_image(img1, 500)
    img2 = resize_image(img2, int(sys.argv[3]))
    petal_shape = get_petal_shape(img1)
    vein_shape = get_vein_shape(img2)

    vein_aligned = align(img1, img2, petal_shape,vein_shape)
    grimg = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    alpha = 0.8
    beta = (1.0 - alpha)
    dst = cv2.addWeighted(grimg, alpha, vein_aligned, beta, 0.0)
    io.imshow(dst)
    io.show()



if __name__ == "__main__":
    main()
