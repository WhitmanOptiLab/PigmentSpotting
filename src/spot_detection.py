import sys
import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import BayesianGaussianMixture
import cv2
import image_shapes
from scipy.stats import norm
from matplotlib.patches import Ellipse
from matplotlib import style
style.use('fivethirtyeight')
import principal_component
import image_utilities as iu

def make_model(X):
    GMM = BayesianGaussianMixture(n_components = 20, max_iter= 300, covariance_type='spherical', verbose=2).fit(X) # Instantiate and fit the model
    print('Converged:', GMM.converged_) # Check if the model has converged
    return GMM

def make_predictions(GMM, d):
    prediction = GMM.predict_proba(d)
    return prediction

def draw_circle(position, radius, ax=None, **kwargs):
    #Draw a circle with a given position and covariance
    ax = ax or plt.gca()
    height = 2 * np.sqrt(radius)
    circ = Ellipse(position, 2*height, 2*height, **kwargs)
    circ.set_facecolor('blue')
    circ.set_edgecolor('blue')
    ax.add_patch(circ)

def draw_circles_on_axis_from_model(ax, m_w_c, i):
    mwc = m_w_c[i]
    w_factor = 0.2 / mwc[2,:].max()
    for x_pos, y_pos, covar, w in zip(mwc[0,:],mwc[1,:],mwc[3,:],mwc[2,:]):
        if w > 0.01:
            draw_circle((x_pos, y_pos), covar, ax=ax, alpha=w * w_factor)

def overlay_spotting_events_on_point_cloud(x, y, m_w_c, i):
    fig = plt.figure(figsize=(5,5))
    ax0 = fig.add_subplot(111)
    ax0.scatter(x,y)
    ax0.invert_yaxis()
    draw_circles_on_axis_from_model(ax0, m_w_c, i)
    plt.show()
    
def draw_circles_on_whole_image(ax, image, m_w_c, left_x, top_y):
    for n, mwc in enumerate(m_w_c):
        w_factor = 0.2 / mwc[2,:].max()
        for x_pos, y_pos, covar, w in zip(mwc[0,:],mwc[1,:],mwc[3,:],mwc[2,:]):
            if w > 0.01:
                draw_circle((x_pos + left_x[n], y_pos  + top_y[n]), covar, ax=ax, alpha=w * w_factor)
    
def overlay_spotting_events_on_image(left_x, top_y, image, m_w_c):
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111)
    ax.imshow(image, cmap='binary_r')
    draw_circles_on_whole_image(ax, image, m_w_c, left_x, top_y)
    plt.show()

    #Pass through PCA grayscale images
def threshold_to_connected_componenets(Pca_image):
    th, _ = cv2.threshold(Pca_image[np.nonzero(Pca_image)], 0, 255, cv2.THRESH_OTSU) #Use otsu algorithm to find appropriate threshold
    _, im_th = cv2.threshold(Pca_image, th, 255, cv2.THRESH_BINARY) #Apply that threshold
    _, _, stats, _ = cv2.connectedComponentsWithStats(im_th, 8, cv2.CV_32S) 
    return th, im_th, stats

    #Omits any recognized componenets that are under 20 pixels in area
def process_componenet_stats(stats):
    too_small = stats[:,4] < 20
    processed_stats = np.delete(stats, np.nonzero(too_small), axis = 0)

    #leftmost_x_coord = processed_stats[1:,0]
    #topmost_y_coord = processed_stats[1:,1]
    #width = processed_stats[1:,2]
    #height = processed_stats[1:,3]
    #area = processed_stats[1:,4]

    return processed_stats

    # Creates a point cloud for the GMM for each spot based on its bounding box
    # then wraps the data into an array
def create_spot_point_clouds(Pca_image, processed_stats, th):
    top_y, bottom_y = processed_stats[1:,1], processed_stats[1:,1] + processed_stats[1:,3]
    left_x, right_x = processed_stats[1:,0], processed_stats[1:,0] + processed_stats[1:,2]
    spot_point_clouds = []
    for i in range(processed_stats.shape[0] - 1):
        X,Y,data = principal_component.create_point_cloud(  Pca_image, 
                                                            top_y[i], 
                                                            bottom_y[i], 
                                                            left_x[i], 
                                                            right_x[i],
                                                            th  )
        row = np.asarray((X,Y,data))
        spot_point_clouds.append(row)
    spot_point_clouds = np.vstack(spot_point_clouds)

    return left_x, top_y, spot_point_clouds

def get_predictions(image, filename, dump_to_file = True, downscaling = 600):
    im = cv2.imread(image)
    im = iu.resize_image(im, int(downscaling))
    petal_shape = image_shapes.get_petal_shape(im)
    pca_image = principal_component.pca_to_grey(im,petal_shape)
    threshold, _ , stats = threshold_to_connected_componenets(pca_image)
    processed_stats = process_componenet_stats(stats)
    left_x, top_y, spot_point_clouds = create_spot_point_clouds(pca_image, processed_stats, threshold)
    m_w_c = []
    for n in range(spot_point_clouds.shape[0]):
        d = np.array([spot_point_clouds[n,0], spot_point_clouds[n,1]]).T 
        model = make_model(d)
        p = make_predictions(model, d)
        m_w_c.append(np.array((model.means_[:,0],model.means_[:,1], model.weights_, model.covariances_)))
    if dump_to_file == True:
        with open('{}.csv'.format(filename), 'a') as file:
            for n, spot in enumerate(m_w_c):
                np.savetxt(file, spot ,delimiter=",",fmt='%f', header = 'spot_{}'.format(n), comments='')
    overlay_spotting_events_on_image(left_x, top_y, pca_image, m_w_c)

    return spot_point_clouds, m_w_c
    

def main():
    if len(sys.argv) == 4:
        get_predictions(sys.argv[1], sys.argv[2], downscaling= sys.argv[3])
    else:
        get_predictions(sys.argv[1], sys.argv[2])
if __name__ == "__main__":
    main()
