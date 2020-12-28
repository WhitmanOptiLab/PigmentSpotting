import sys
import os.path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import BayesianGaussianMixture
from cv2 import imread, cvtColor, COLOR_BGR2RGB
from scipy.stats import norm
from matplotlib.patches import Ellipse
from matplotlib import style
style.use('fivethirtyeight')

import principal_component
import image_utilities as iu

def make_model(X):
    GMM = BayesianGaussianMixture(n_components=200, max_iter=700, covariance_type='spherical', verbose=2).fit(X) # Instantiate and fit the model
    print('Converged:',GMM.converged_) # Check if the model has converged
    return GMM

def make_predictions(GMM, d):
    prediction = GMM.predict_proba(d)

def draw_circle(position, radius, ax=None, **kwargs):
    """Draw a circle with a given position and covariance"""
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

def main():
    if not os.path.exists(sys.argv[1]): 
        raise ValueError("The image files doesn't Exists. Please review your file path")
    image = imread(sys.argv[1])
    image = iu.resize_image(image, 400)

    x, y, im = principal_component.create_point_cloud(image)
    d = np.array([x,y]).T
    model = make_model(d)

    p = make_predictions(model, d)

    overlay_spotting_events_on_point_cloud(x, y, model, im)
    overlay_spotting_events_on_image(cvtColor(image, COLOR_BGR2RGB), model)


if __name__ == "__main__":
    main()
