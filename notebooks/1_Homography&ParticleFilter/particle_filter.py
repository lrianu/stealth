#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 16 21:08:51 2018
Particle Filter.  Particle filtering is not implemented in openCV 3.  Use openCV2 environment.

@author: lrianu
"""

import cv2
import sys

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

if __name__ == '__main__':

    # Set up tracker
    # Instead of MIL, you can also use
    
    tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']   # 'GOTURN' crashes in OpenCV 3.4.1
    tracker_type = tracker_types[4]

    if int(minor_ver) < 3:
        tracker = cv2.Tracker_create(tracker_type)
    else:
        # Particle Filter is not yet implemented in OpenCV 3.
        # TODO: Print a warning and return Error if using OpenCV 3  

# Set up tracker
tracker = cv2.TrackerSampler.TrackerSampler()

# Read video
    video = cv2.VideoCapture('FishVideo.mp4')

    # Exit if video not opened.
    if not video.isOpened():
        print('Could not open video')
        sys.exit()

    # Read first frame.
    ok, frame = video.read()
    if not ok:
        print('Cannot read video file')
        sys.exit()

    # Define an initial bounding box
    # bbox = (250, 450, 200, 470)     # TODO: check if [r, h, c, w] is correct format.

    # Uncomment the line below to select a different bounding box
    bbox = cv2.selectROI(frame, False)