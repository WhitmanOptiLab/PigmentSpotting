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

def draw_circles_on_axis_from_model(ax, model):
    w_factor = 0.2 / model.weights_.max()
    for pos, covar, w in zip(model.means_, model.covariances_, model.weights_):
        if w > 0.01:
            draw_circle(pos, covar, ax=ax, alpha=w * w_factor)

def overlay_spotting_events_on_point_cloud(x, y, model, im):
    fig = plt.figure(figsize=(5,5))
    ax0 = fig.add_subplot(111)
    ax0.scatter(x,y)
    ax0.invert_yaxis()
    draw_circles_on_axis_from_model(ax0, model)
    plt.show()

def overlay_spotting_events_on_image(image, model):
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    draw_circles_on_axis_from_model(ax, model)
    plt.show()

    #Pass through PCA grayscale images
def threshold_to_connected_componenets(Pca_image):
    th, _ = cv2.threshold(Pca_image[np.nonzero(Pca_image)], 0, 255, cv2.THRESH_OTSU)
    _, im_th = cv2.threshold(Pca_image, th, 255, cv2.THRESH_BINARY)
    _, _, stats, _ = cv2.connectedComponentsWithStats(im_th, 8, cv2.CV_32S)
    return th, im_th, stats

    #Omits any recognized componenets that are under 20 pixels in area
def process_componenet_stats(stats):
    too_small = stats[:,4] < 20
    processed_stats = np.delete(stats, np.nonzero(too_small), axis = 0)

    #leftmost_x_coord = new_stats[1:,0]
    #topmost_y_coord = new_stats[1:,1]
    #width = new_stats[1:,2]
    #height = new_stats[1:,3]
    #area = new_stats[1:,4]

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

    return spot_point_clouds


def main():
    image = cv2.imread(sys.argv[1])
    petal_shape = image_shapes.get_petal_shape(image)
    pca_image = principal_component.pca_to_grey(image,petal_shape)
    threshold, threshold_image, stats = threshold_to_connected_componenets(pca_image)
    processed_stats = process_componenet_stats(stats)
    spot_point_clouds = create_spot_point_clouds(pca_image, processed_stats, threshold)
    predictions = []
    for n in range(spot_point_clouds.shape[0]):
        d = np.array([spot_point_clouds[n,0], spot_point_clouds[n,1]]).T 
        model = make_model(d)
        p = make_predictions(model, d)
        predictions.append(d)
        overlay_spotting_events_on_point_cloud( spot_point_clouds[n,0], 
                                                spot_point_clouds[n,1], 
                                                model, 
                                                spot_point_clouds[n,2])
        overlay_spotting_events_on_image(spot_point_clouds[n,2], model)

if __name__ == "__main__":
    main()