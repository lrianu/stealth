#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Find the 25 best corners to track in the image using Shi-Tomasi method

Created on Fri Jun  8 18:28:52 2018

@author: lrianu
"""

# Find the 25 best corners to track in two images using Shi-Tomasi method
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img1 = cv.imread('/Users/lrianu/Desktop/snapshots/image-088.png')
img2 = cv.imread('/Users/lrianu/Desktop/snapshots/image-089.png')

# Use Shi-Tomasi keypoints to compute homography via RANSAC
gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

corners1 = cv.goodFeaturesToTrack(gray1, 25, 0.01, 10)
corners2 = cv.goodFeaturesToTrack(gray2, 25, 0.01, 10)

M, mask = cv.findHomography(corners1, corners2, cv.RANSAC, 5.0)
# matchesMask = mask.ravel().tolist()

# Find the Fundamental Matrix
F, mask = cv.findFundamentalMat(corners1, corners2, cv.FM_RANSAC)

# TODO: This has so much noise it returns a rank 3.  Refine F.

matchesMask = mask.ravel().tolist()

# Convert corners to keypoints
x1 = corners1[:, :, 0]
y1 = corners1[:, :, 1]
x2 = corners2[:, :, 0]
y2 = corners2[:, :, 1]
kp1 = [cv.KeyPoint(x1[idx], y1[idx], 1) for idx in range(len(x1))]
kp2 = [cv.KeyPoint(x2[idx], y2[idx], 1) for idx in range(len(x2))]

# Visualize corners in image
corners1 = np.int0(corners1)
corners2 = np.int0(corners2)

for i in corners1:
    x, y = i.ravel()
    cv.circle(img1, (x, y), 3, 255, -1)

for j in corners2:
    u, v = j.ravel()
    cv.circle(img2, (u, v), 3, 255, -1)

fig1 = plt.figure(figsize=(18, 16), dpi=80, facecolor='w', edgecolor='k')
plt.imshow(img1), plt.show()

fig2 = plt.figure(figsize=(18, 16), dpi=80, facecolor='w', edgecolor='k')
plt.imshow(img2), plt.show()
