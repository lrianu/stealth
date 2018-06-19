#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jun 18 14:13:53 2017

@author: lrianu
"""

import cv2
import numpy as np
import kullback_lieber_divergence as KL

_img = []  # stores clean image
_img_dirty = []  # stores image to draw rectangle during cropping
_points = []  # stores the points for the cropping


def calc_histogram(patch):
    """ Color histogram is calculated and the 3 histograms
    are appended into a one-dimensional vector and normalized. """
    # TODO: Try with HSV color space.  This is generally more robust to 
    # variations in brightness.
    
    # Hyperparameters
    num_bins = 32

    # used to calculate histogram for a part of the image
    # but extract_patch() is used instead of this.
    mask = None

    blue_model = cv2.calcHist([patch], [0], mask, [num_bins],
                              [0, 256]).flatten()
    green_model = cv2.calcHist([patch], [1], mask, [num_bins],
                               [0, 256]).flatten()
    red_model = cv2.calcHist([patch], [2], mask, [num_bins],
                             [0, 256]).flatten()

    color_patch = np.concatenate((blue_model, green_model, red_model))

    # Normalize histogram values for the KL divergence computation
    color_patch = color_patch/np.sum(color_patch)
    return color_patch


def extract_patch(s_t, image, object_bound):
    """ Extracts the part of the image corresponding to a given state. """
    s_t = s_t.astype(int)  # Convert state to discrete pixel locations
    return image[s_t[0]:s_t[0] + object_bound[0],
                 s_t[1]:s_t[1] + object_bound[1]]


def appearance_model(s_t, image, color_model, object_bound):
    """ This is the model of what the object should look like
    and is used to check how different each particle is from
    the model. In this case, a color histogram is used. The
    difference between the histograms is determined by the KL
    divergence and the likelihood of it being a match is given
    by the exponential distribution. Lambda is a hyperparameter
    that determines how strict the model is on how close the
    particle and the model have to be to be a likely match.
    If the appearance of the object can change significantly,
    consider reducing this value. It can also help to not
    prematurely converge on the wrong object when the particles
    are initally scattered. """
    # Hyperparameters
    l = 1  # lambda

    patch = extract_patch(s_t, image, object_bound)
    color_patch = calc_histogram(patch)

    divergence = KL.discrete_kl_divergence(color_model, color_patch)

    likelihood = np.exp(-l*divergence)
    return likelihood


def motion_model(s_t, image_bound, object_bound):
    """ This is the model of how we expect the object to move between
    consecutive frames. We use a normal ditribution to model this.
    In this case, we assume the particle will move with a 95%
    certainty within a radius of 2*std_dev pixels. If the object is
    likely to move faster, then you should increase this value. In other
    problems like localization. The motion direction could be based
    on a noisy spedometer value and the distibution would have a
    non-zero mean added to the particle's current location. """
    # Hyperparameters
    """ Set this large if the object moves fast/is closer to the camera.
    For face tracking, you might need a std_dev of 40 but for security
    camera footage the distance from the object being tracked is greater
    and has a smaller speed in terms of pixels/sec. """
    std_dev = 2

    # Motion estimation
    s_t = s_t + std_dev * np.random.randn(s_t.shape[0], 2)

    # Out-of-bounds check
    s_t[:, 0] = np.maximum(0, np.minimum(image_bound[0]-object_bound[0],
                           s_t[:, 0]))
    s_t[:, 1] = np.maximum(0, np.minimum(image_bound[1]-object_bound[1],
                           s_t[:, 1]))
    return s_t


def random_sample(w_t):
    """ Function to resample the particles according to their weights, i.e.
    points that have a higher weight have a higher probability of being chosen
    again. The sampling is done with replacement so that unlikely particles
    will be removed and good ones will reproduce. It's basically a form of
    natural selection. An improvement would be to always have some small
    chance of 'mutation' to occur that is that with some small value epsilon
    the point can be a uniform randomly selected point on the grid. This will
    enable exploration in case the object is lost. """
    cumsum = np.cumsum(w_t)
    draws = np.random.rand(w_t.shape[0])
    idxs = np.zeros(w_t.shape[0])
    for i, draw in enumerate(draws):
        for j, probability in enumerate(cumsum):
            if probability > draw:
                idxs[i] = j
                break
    return idxs.astype(int)


def crop(event, x, y, flags, param):
    """ This is the mouse event listener function to determine
    the cropping points for the object of interest. """

    global _img_dirty
    # get clean copy of the image without previous rectangle
    _img_dirty = _img.copy()
    if event == cv2.EVENT_LBUTTONUP:
        # Add point to the vector
        _points.append((x, y))
    elif event == cv2.EVENT_MOUSEMOVE and _points != []:
        # Draws the rectangle while cropping
        cv2.rectangle(_img_dirty, _points[0],
                      (x, y), (0, 255, 0), 2)


def get_roi(cap):
    """ This function allows the user to select the object
    they want to track. """

    # Get a frame for cropping
    while True:
        ret, img = cap.read()

        img_cpy = img.copy()  # copy for drawing on
        cv2.putText(img_cpy, "Press P to Pause", (15, 30),
                    cv2.QT_FONT_NORMAL, 1, (255, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow("Press P to pause", img_cpy)
        if cv2.waitKey(30) & 0xFF == ord('p'):
            break

    cv2.destroyAllWindows()

    global _img, _img_dirty
    _img = img.copy()  # global clean copy
    cv2.putText(_img, "Press R to Reset Selection", (15, 30),
                cv2.QT_FONT_NORMAL, 1, (255, 0, 0), 2, cv2.LINE_AA)

    _img_dirty = _img.copy()

    cv2.namedWindow("Select Region")
    cv2.setMouseCallback("Select Region", crop)

    # Wait till the crop consists of 2 points
    global _points
    while (len(_points) != 2):
        cv2.imshow("Select Region", _img_dirty)
        key = cv2.waitKey(30) & 0xFF
        if key == ord('r'):
            # reset the image to the clean image delete current points
            _points = []
            _img_dirty = _img.copy()

    cv2.destroyAllWindows()

    tmp_img = img.copy()  # copy the image to draw the rectangle on it.

    # Determine top-left and bottom-right corners of the rectangle
    top_left = min(_points[0][0], _points[1][0]), min(_points[0][1],
                                                      _points[1][1])
    bottom_right = max(_points[0][0], _points[1][0]), max(_points[0][1],
                                                          _points[1][1])

    # Crop out the object of interest
    object_of_interest = img[top_left[1]:bottom_right[1],
                             top_left[0]:bottom_right[0]]

    # Show the selected region
    cv2.rectangle(tmp_img, top_left, bottom_right, (0, 255, 0), 2)
    cv2.imshow("Selected Region", tmp_img)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()

    return object_of_interest


def main(v):

    # Hyperparameters
    NUM_PARTICLES = 200

    # Setting the seed for the random generator to
    # create predictable executions during testing
    np.random.seed(0)

    cap = cv2.VideoCapture(v)

    # get a frame for its dimensions
    ret, img = cap.read()
    image_bound = img.shape

    # Get the object to track.
    object_of_interest = get_roi(cap)
    object_bound = object_of_interest.shape

    # Calculate the appearance model of the object.
    color_model = calc_histogram(object_of_interest)

    """ Uniform randomly scatter the particles. You could use the cropping
     location but assume you know already what object you want to track
     but not where it is initially in the image then scattering the particles
     will enable you to quickly scan the image."""
    # The state is (y,x) coordinates of the top-left corner of the rectangle
    s_t = np.random.rand(NUM_PARTICLES, 2)
    s_t[:, 0] *= image_bound[0]
    s_t[:, 1] *= image_bound[1]

    w_t = np.ones(NUM_PARTICLES)/NUM_PARTICLES  # Initially equal weights

    likelihood = np.zeros(NUM_PARTICLES)

    # Default draw settings
    draw_particles = True
    draw_info = True
    # For each frame of the video
    while True:
        # Load current frame
        ret, img = cap.read()

        """ PARTICLE FILTER LOOP """
        # Randomly sample particles according to weights
        idxs = random_sample(w_t)
        s_t = s_t[idxs, :]
        w_t = w_t[idxs]

        # Move particles according to motion model
        s_t = motion_model(s_t, img.shape, object_bound)

        # Compute appearance likelihood for each particle
        for j in range(NUM_PARTICLES):
            likelihood[j] = appearance_model(s_t[j, :], img,
                                             color_model, object_bound)

        # Update particle weights
        w_t = w_t*likelihood
        w_t = w_t/np.sum(w_t)

        # Estimate object location based on weighted
        # states of the particles.
        estimate_t = (s_t.T.dot(w_t)).astype(int)

        """ DRAWING """
        # Draw box around the mean estimate
        cv2.rectangle(img, (estimate_t[1], estimate_t[0]),
                      (estimate_t[1]+object_bound[1],
                       estimate_t[0]+object_bound[0]),
                      (0, 255, 0), 2)
        # Draw the particles
        if draw_particles:
            for j in range(NUM_PARTICLES):
                # Draw particle in center of the rectangle
                # that its state is dictating.
                cv2.circle(img, (s_t[j, 1].astype(int)+int(object_bound[1]/2),
                                 s_t[j, 0].astype(int)+int(object_bound[0]/2)),
                           5, (255, 0, 0))

        # Draw the instructions
        if draw_info:
            cv2.putText(img, "Press T to Toggle Particles", (15, 30),
                        cv2.QT_FONT_NORMAL, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(img, "Press Q to Quit", (15, 60),
                        cv2.QT_FONT_NORMAL, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(img, "Press I to Toggle Info", (15, 90),
                        cv2.QT_FONT_NORMAL, 0.5, (255, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow("Press Q to Quit", img)
        key = cv2.waitKey(30) & 0xFF
        # Handle key presses
        if key == ord('q'):
            break
        elif key == ord('t'):
            draw_particles = not draw_particles  # toggle
        elif key == ord('i'):
            draw_info = not draw_info

    # Release resources and close all windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    video = 'FishVideo.mp4'
    main(video)
