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
