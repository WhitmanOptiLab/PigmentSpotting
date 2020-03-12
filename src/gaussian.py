import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import BayesianGaussianMixture
from cv2 import imread
from scipy.stats import norm
from matplotlib.patches import Ellipse
from matplotlib import style
style.use('fivethirtyeight')

import principlecomponent
import imageutilities as iu

def make_model(X):
    GMM = BayesianGaussianMixture(n_components=100, covariance_type='spherical', verbose=2).fit(X) # Instantiate and fit the model
    print('Converged:',GMM.converged_) # Check if the model has converged
    return GMM

def make_predictions(GMM, d):
    prediction = GMM.predict_proba(d)

def draw_circle(position, radius, ax=None, **kwargs):
    """Draw a circle with a given position and covariance"""
    ax = ax or plt.gca()

    height = 2 * np.sqrt(radius)

    circ = Ellipse(position, 2*height, 2*height, **kwargs)
    circ.set_facecolor('red')
    circ.set_edgecolor('red')

    ax.add_patch(circ)

def make_plot(x, y, gmm, im):
    fig = plt.figure(figsize=(5,5))
    ax0 = fig.add_subplot(111)
    ax0.scatter(x,y)
    ax0.invert_yaxis()
    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        if w > 0.01:
            draw_circle(pos, covar, alpha=w * w_factor)

    plt.show()

def main():
    image = imread(sys.argv[1])
    image = iu.resize_image(image, 350)
    x,y, im = principlecomponent.create_point_cloud(image)
    d = np.array([x,y]).T
    # xy = np.meshgrid(x,y)
    model = make_model(d)
    p = make_predictions(model,d)
    make_plot(x,y,model, im)


if __name__ == "__main__":
    main()
