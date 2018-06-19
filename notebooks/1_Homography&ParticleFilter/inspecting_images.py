#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Read in and visually inspect a single still image, checking for data types and 
ranges

Created on Fri Jun  8 14:47:48 2018

@author: lrianu
"""

#%matplotlib inline (for IPython console)

import matplotlib.pyplot as plt
from IPython.display import Image
import cv2
import numpy as np

#Let the system know which image you want to view
img_fname = '/Users/lrianu/Desktop/snapshots/image-087.png'

#Read image
img = cv2.imread(img_fname, cv2.IMREAD_UNCHANGED)

#View the image (for IPython console)
Image(filename=img_fname)

##View image (for Python console)
#cv2.imshow('Sample Image', img)
#
#cv2.waitKey(0) #waits until a key is pressed
#cv2.destroyAllWindows() #destroys the window showing image

#Get dimensions of image
dimensions = img.shape

#Height, width, number of channels in image
height = img.shape[0]
width = img.shape[1]
channels = img.shape[2]

print('Image Dimension     : ', dimensions)
print('Image Height        : ', height)
print('Image Width         : ', width)
print('Number of Channels  : ', channels)

#Image data type, pixel range
dtype = img.dtype
pixelmin = np.amin(img)
pixelmax = np.amax(img)

print('Image Data Type     : ', dtype)
print('Image Pixel Range   : ', [pixelmin, pixelmax])

#Inspect a 1D row of pixel intensities
x = np.arange(width)
y = img[int(height/2)][:]   #multichannel will plot one line per channel
plt.plot(x, y)

#Convert image to float and normalize to zero mean, range [0 1]
norm_image = cv2.normalize(img, None, alpha=0, beta=1,
                           norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

#Inspect a 1D row of pixel intensities
x = np.arange(width)
y = norm_image[int(height/2)][:]   #multichannel will plot one line per channel
plt.plot(x, y)

#Inspect origin of image
plt.imshow(cv2.cvtColor(norm_image, cv2.COLOR_BGR2RGB))   

    #TODO: whenever read in with cv2.imread and out with plot.imshow, must convert GRB2RGB
    #TODO: whenever display with plt.imshow, clips float to [0 1]

#Inspect color channels
b,g,r = cv2.split(img)



