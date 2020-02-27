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




def make_model(X):
    GMM = GaussianMixture(n_components=4).fit(X) # Instantiate and fit the model
    print('Converged:',GMM.converged_) # Check if the model has converged
    return GMM

def make_predictions(GMM, d):
    prediction = GMM.predict_proba(d)
    return prediction

def make_data(d):
    x,y = np.meshgrid(np.sort(d[:,0]),np.sort(d[:,1]))
    XY = np.array([x.flatten(),y.flatten()]).T
    return XY
def make_plot(X, XY, GMM):
    fig = plt.figure(figsize=(5,5))
    ax0 = fig.add_subplot(111)
    ax0.scatter(X[:,0],X[:,1])
    #ax0.scatter(Y[0,:],Y[1,:],c='orange',zorder=10,s=100)
    for m,c in zip(GMM.means_,GMM.covariances_):
        multi_normal = multivariate_normal(mean=m,cov=c)
        ax0.contour(np.sort(X[:,0]),np.sort(X[:,1]),multi_normal.pdf(XY).reshape(len(X),len(X)),colors='black',alpha=0.3)
        ax0.scatter(m[0],m[1],c='grey',zorder=10,s=100)
    plt.show()

def main():
    image = cv2.imread(sys.argv[1])
    d = principlecomponent.create_point_cloud(image)
    xy = make_data(d)
    plt.imshow(d)
    plt.show()
    model = make_model(d)
    p = make_predictions(model,d)
    make_plot(d,xy,model)

if __name__ == "__main__":
    main()
