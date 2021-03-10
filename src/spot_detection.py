import os
import sys
import cv2
import time 
import image_shapes
import principal_component


import numpy as np
import image_utilities as iu
import matplotlib.pyplot as plt

from scipy.stats import norm
from matplotlib import style
style.use('fivethirtyeight')
from skimage.metrics import mean_squared_error
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal
from sklearn.mixture import BayesianGaussianMixture



def make_model(X, n_components):
    # Instantiates and fits the model
    GMM = BayesianGaussianMixture(n_components = n_components, max_iter= 300, covariance_type='spherical', verbose=0).fit(X) 
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
    
def overlay_spotting_events_on_image(left_x, top_y, image, m_w_c, save_to_file = False):

    # Takes the whole image as well as the means, weights, and co-variances 
    # and uses the coordinates of each spots bounding box to overlay each gaussian onto the whole in the correct place

    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111)
    ax.imshow(image, cmap='binary_r')
    draw_circles_on_whole_image(ax, image, m_w_c, left_x, top_y)
    plt.show()
    
def threshold_to_connected_componenets(Pca_image):
    #Pass through PCA grayscale images

    th, _ = cv2.threshold(Pca_image[np.nonzero(Pca_image)], 0, 255, cv2.THRESH_OTSU) #Use otsu algorithm to find appropriate threshold
    _, im_th = cv2.threshold(Pca_image, th + 10, 255, cv2.THRESH_BINARY) #Apply that threshold 
    _, _, stats, _ = cv2.connectedComponentsWithStats(im_th, 8, cv2.CV_32S) 
    return th, im_th, stats

def process_componenet_stats(stats):

    # Takes the stats array given by the connected components and removes any componenet smaller than 20 pixels
    # Also removes the first component which is the background

    too_small = stats[:,4] < 20
    processed_stats = np.delete(stats, np.nonzero(too_small), axis = 0)
    processed_stats = np.delete(processed_stats, (0), axis=0)

    #leftmost_x_coord = processed_stats[1:,0]
    #topmost_y_coord = processed_stats[1:,1]
    #width = processed_stats[1:,2]
    #height = processed_stats[1:,3]
    #area = processed_stats[1:,4]

    return processed_stats

def create_spot_point_clouds(Pca_image, processed_stats, th):

    # Creates a point cloud for the GMM for each spot based on its bounding box
    # Then wraps the data into an array

    top_y, bottom_y = processed_stats[:,1], processed_stats[:,1] + processed_stats[:,3]
    left_x, right_x = processed_stats[:,0], processed_stats[:,0] + processed_stats[:,2]
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

    return left_x, right_x, top_y, bottom_y, spot_point_clouds

def get_gaussians_image(processed_stats_for_spot , m_w_c):

    # Creates a mesh-grid that is the same size of the spot image using the stats from connected components
    # Then uses the means, weights, and co-variances from the GMM model to overlay each guassian in its respective spot
    # Returns an image of every gaussian for the respective spot

    x,y = np.mgrid[ 0:processed_stats_for_spot[3]:processed_stats_for_spot[3] * 1j, 0:processed_stats_for_spot[2]:processed_stats_for_spot[2] * 1j]
    pos = np.dstack((x,y))
    weight_factors = []
    gaussians = []
    for x_pos, y_pos, w, covar in zip(m_w_c[0,:],m_w_c[1,:],m_w_c[2,:],m_w_c[3,:]):
        gaussian = multivariate_normal([y_pos,x_pos],[[covar,0],[0,covar]])
        gaussian = gaussian.pdf(pos)
        gaussian = np.interp(gaussian,(np.min(gaussian), np.max(gaussian)), ((0,255)))
        gaussians.append(gaussian)
    gaussians = np.asarray(gaussians)
    gaussians_image = np.empty_like(gaussians[0])
    for gaussian in gaussians:
        gaussians_image = np.maximum(gaussians_image, gaussian, out=gaussians_image)
    return gaussians_image

def rmsdiff(im1, im2):
    return np.sqrt(mean_squared_error(im1.astype('float'), im2.astype('float')))

def mask_spot_image(top_y_for_spot, bottom_y_for_spot, left_x_for_spot, right_x_for_spot, threshold_image, spot_image):
    threshold_image_at_spot = threshold_image[top_y_for_spot:bottom_y_for_spot,left_x_for_spot:right_x_for_spot]
    return spot_image * threshold_image_at_spot

def save_to_file(spot_results, filename):
    if os.path.exists('{}.csv'.format(filename)):
        os.remove('{}.csv'.format(filename))
    with open('{}.csv'.format(filename), 'a') as file:
        for n, spot in enumerate(spot_results):
            np.savetxt(file, spot.T,delimiter=",",fmt='%f', header = 'spot_{}'.format(n+1))

def get_predictions(im, filename, dump_to_file = True, rms_threshold = 70, downscaling = 600):
    #tic = time.time()
    im_downscaled = iu.resize_image(im, int(downscaling)) # Downscaling the image   
    x_rescale_factor = im.shape[0]/im_downscaled.shape[0] # The rescale factors are used to rescale the results
    y_rescale_factor = im.shape[1]/im_downscaled.shape[1] # from the GMM back to the coordinate system of the original image 
    petal_shape = image_shapes.get_petal_shape(im_downscaled)
    pca_image = principal_component.pca_to_grey(im_downscaled,petal_shape)
    threshold, threshold_image , stats = threshold_to_connected_componenets(pca_image)
    processed_stats = process_componenet_stats(stats)
    left_x, right_x, top_y, bottom_y, spot_point_clouds = create_spot_point_clouds(pca_image, processed_stats, threshold) 
    spot_results = [] 
    for i , spot in enumerate(spot_point_clouds):
        print("======= SPOT # {} =======".format(i))

        # Make an array using the point cloud data for the spot 
        # Initialize the model with one component on said array

        m_w_c = []
        rmsdiff_improvement = 0
        d = np.array([spot[0], spot[1]]).T 
        model = make_model(d,1)
        p = make_predictions(model, d)
        m_w_c.append(np.array((model.means_[:,0],model.means_[:,1], model.weights_, model.covariances_)))
        m_w_c = np.hstack(m_w_c)
        gaussians_image = get_gaussians_image(processed_stats[i,:], m_w_c)
        n_components = 1
        m_w_c_prev = m_w_c
        bias = (np.log(np.average(processed_stats[:,4])) -  np.log(processed_stats[i,4])) # A bias is added to the RMS difference threshold. ~ -5 for the largest spots and ~ 5 for the smallest
        masked_image = mask_spot_image(top_y[i],bottom_y[i],left_x[i],right_x[i],threshold_image,spot[2])
        #print("______________RMSDIFF____{}".format(rmsdiff(masked_image, gaussians_image)))
        while rmsdiff(masked_image, gaussians_image) > int(rms_threshold) + bias:
            
            # Continue to add componenets to the model until the RMS difference reaches an approriate threshhold
            
            print("# OF COMPONENTS: {}".format(n_components))
            m_w_c = []
            n_components += 1
            cost = 2.5/n_components
            model = make_model(d,n_components)
            p = make_predictions(model, d)
            m_w_c.append(np.array((model.means_[:,0],model.means_[:,1], model.weights_, model.covariances_)))
            m_w_c = np.hstack(m_w_c)
            gaussians_image_prev = get_gaussians_image(processed_stats[i,:], m_w_c_prev)
            gaussians_image = get_gaussians_image(processed_stats[i], m_w_c)
            rmsdiff_improvement = rmsdiff(masked_image, gaussians_image)/rmsdiff(masked_image, gaussians_image_prev)
            #print("CURRENT_COST_FOR_{}_COMPONENTS: {}".format(n_components, cost))
            #print("RMS_IMPROVEMENT_FOR_{}_COMPONENTS: {}".format(n_components, rmsdiff_improvement))

            # 

            if rmsdiff_improvement > cost: 
                m_w_c = m_w_c_prev 
                break
            else:
                m_w_c_prev = m_w_c 
        m_w_c[0] = (m_w_c[0] * x_rescale_factor)
        m_w_c[1] = (m_w_c[1] * y_rescale_factor)
        m_w_c[2] = (m_w_c[2] * y_rescale_factor * x_rescale_factor)
        m_w_c[3] = (m_w_c[3] * y_rescale_factor * x_rescale_factor)
        spot_results.append(m_w_c)
    #toc = time.time()
    #print("Time to run: {} seconds".format(toc - tic))

    if dump_to_file == True:
        save_to_file(spot_results, filename)
    
    petal_shape_original_size = image_shapes.get_petal_shape(im)
    pca_image_original_size = principal_component.pca_to_grey(im, petal_shape_original_size)
    overlay_spotting_events_on_image((left_x * x_rescale_factor), (top_y * y_rescale_factor), pca_image_original_size, [spot_results[i] for i in range(len(spot_results))])

    return pca_image_original_size, spot_results
    

def main():
    image = cv2.imread(sys.argv[1])
    if len(sys.argv) < 3:
        print("Image path or filename not present \n please put [image name] [filename]")
    elif len(sys.argv) == 4:
        get_predictions(image, sys.argv[2], rms_threshold=sys.argv[3]) 
    elif len(sys.argv) == 5:
        get_predictions(image, sys.argv[2], rms_threshold=sys.argv[3], downscaling=sys.argv[4])
    else:
        get_predictions(image, sys.argv[2])

if __name__ == "__main__":
    main()
