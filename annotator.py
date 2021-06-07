#!/usr/bin/env python
# coding: utf-8

# In[23]:


import numpy as np 
import cv2 


# In[24]:


# global variables: 
x_coordinate, y_coordinate = 0, 0
points = [] 
downscale_percent = 0.75
upscale_percent = 1/0.75


# In[25]:


def scale_img(img, scaling_percent):
    width, height =  int(img.shape[0]*scaling_percent), int(img.shape[1]*scaling_percent)
    dimensions = (height, width)
    scaled_img = cv2.resize(img, dimensions, interpolation = cv2.INTER_AREA)
    return scaled_img
    


# In[26]:


def detect_centeral_vein(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        x_coordinate, y_coordinate = int(x * upscale_percent), int(y*upscale_percent)
        points.append((x_coordinate, y_coordinate))
        print(points)
        print(x,y)
        
   


# In[27]:


img = cv2.imread('.\images\F1P101_Vein_Center_200724.jpg')
dimg = scale_img(img, downscale_percent)
cv2.imshow("Selected Image", dimg)

cv2.setMouseCallback("Selected Image", detect_centeral_vein)
cv2.waitKey(0)
cv2.destroyAllWindows()
# Debugging 
for center in points:
    center_coordinates, radius, color, thickness = center, 2, (255,255,255), 3
    img = cv2.circle(img, center_coordinates, radius, color, thickness)
    
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# We should have a tool that can show the image and its annotations. 


# In[ ]:





# In[ ]:




