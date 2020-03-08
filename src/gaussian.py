import matplotlib.pyplot as plt
import principlecomponent
import sys
from matplotlib import style
style.use('fivethirtyeight')
from sklearn.datasets.samples_generator import make_blobs
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
import cv2
from scipy.stats import norm
from numpy import linalg as LA
import imageutilities as iu
from matplotlib.patches import Ellipse
def make_model(X):
    GMM = GaussianMixture(n_components=35).fit(X) # Instantiate and fit the model
    print('Converged:',GMM.converged_) # Check if the model has converged
    return GMM

def make_predictions(GMM, d):
    prediction = GMM.predict_proba(d)

def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()

    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)

    
    circ = Ellipse(position, 2*height, 2*height, angle, **kwargs)
    circ.set_facecolor('red')
    circ.set_edgecolor('red')

    ax.add_patch(circ)

def make_plot(x, y, gmm, im):
    fig = plt.figure(figsize=(5,5))
    ax0 = fig.add_subplot(111)
    ax0.scatter(x,y)
    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor)

    plt.show()

def main():
    image = cv2.imread(sys.argv[1])
    image = iu.resize_image(image, 350)
    x,y, im = principlecomponent.create_point_cloud(image)
    d = np.array([x,y]).T
    # xy = np.meshgrid(x,y)
    model = make_model(d)
    p = make_predictions(model,d)
    make_plot(x,y,model, im)


if __name__ == "__main__":
    main()
