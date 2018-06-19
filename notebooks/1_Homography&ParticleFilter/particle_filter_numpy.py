#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 12:40:54 2018

reference: https://github.com/prateekroy/Computer-Vision/blob/master/HW3/
detection_tracking.py

@author: lrianu
"""
# import sys
import cv2
import numpy as np

# Read the video file
video = cv2.VideoCapture('FishVideo.mp4')
file_name = 'output_particle_tracker.txt'


def particle_tracker(v, file_name):
    # Open output file
    # output_name = sys.argv[3] + file_name
    # output = open(output_name,"w")
    output = open(file_name, 'w')   # cd to output directory

    frameCounter = 0
    # read first frame
    ret, frame = v.read()
    if ret is False:
        return

    # detect fish in first frame
    x, y, w, h = 450, 250, 470, 200    # TODO: allow to select bounding box

    # Write track point for first frame
    pt = (frameCounter, x+w/2, y+h/2)
    output.write("%d,%d,%d\n" % pt)  # Write as 0,pt_x,pt_y
    frameCounter = frameCounter + 1

    # set the initial tracking window
    track_window = (x, y, w, h)

    # NOTE: you do not need this in the Kalman, Particle or OF trackers
    roi_hist = hsv_histogram_for_window(frame, (x, y, w, h))

    # hist_bp: obtain using cv2.calcBackProject and the HSV histogram
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist_bp = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

    # Spread 300 random particles near object to track
    n_particles = 300

    init_pos = np.array([x+w/2.0, y + h/2.0], int)  # Initial position

    # Init particles to init position
    particles = np.ones((n_particles, 2), int) * init_pos

    # Evaluate appearance model
    # f0 = particleevaluator(hist_bp, pos) * np.ones(n_particles)

    # weights are uniform (at first)
    weights = np.ones(n_particles) / n_particles

    while(1):
        ret, frame = v.read()  # read another frame
        if ret is False:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist_bp = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
        # cv2.imshow('hist_bp', hist_bp)

        # perform the tracking
        stepsize = 18    # TODO: change this from hard-coded number

        # Particle motion model: uniform step
        # TODO: find a better motion model
        np.add(particles,
               np.random.uniform(-stepsize, stepsize, particles.shape),
               out=particles,
               casting="unsafe")

        # Clip out-of-bounds particles
        particles = particles.clip(np.zeros(2),
                                   np.array((frame.shape[1],
                                             frame.shape[0]))-1).astype(int)

        f = particle_evaluator(hist_bp, particles.T)  # Evaluate particles

        # Try to show some visuals
        for i in range(len(f)):
            if f[i] >= 1:
                # Good Particles
                draw_circle(frame, particles[i].T, 1, (0, 0, 255))
            else:
                # Bad Particles
                draw_circle(frame, particles[i].T, 1, (0, 0, 0))

        # Weight ~ histogram response #clip all bad particles
        weights = np.float32(f.clip(1))
        weights /= np.sum(weights)  # Normalize w

        # Expected position: weighted average
        pos = np.sum(particles.T * weights, axis=1).astype(int)

        draw_cross(frame, (np.int32(pos[0]), np.int32(pos[1])), (0, 255, 0), 3)

        # If particle cloud degenerate:
        if 1. / np.sum(weights**2) < n_particles / 2.:
            # Resample particles according to weights
            particles = particles[resample(weights), :]

        cv2.imshow('frame', frame)
        pt = (frameCounter, np.int32(pos[0]), np.int32(pos[1]))
        output.write("%d,%d,%d\n" % pt)  # Write as frame_index,pt_x,pt_y
        frameCounter = frameCounter + 1
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

    output.close()


def hsv_histogram_for_window(frame, window):
    # TODO: check these hard coded numbers
    # set up the ROI for tracking
    c, r, w, h = window
    roi = frame[r:r+h, c:c+w]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi,
                       np.array((0., 60., 32.)),
                       np.array((180., 255., 255.)))
    roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
    return roi_hist


def particle_evaluator(back_proj, particle):
    return back_proj[particle[1], particle[0]]


def draw_circle(img, center, radius, color):
    img = cv2.circle(img, (center[0], center[1]), radius, color, -1)


def draw_cross(img, center, color, d):
    cv2.line(img,
             (center[0] - d, center[1] - d), (center[0] + d, center[1] + d),
             color, 1, cv2.LINE_AA, 0)
    cv2.line(img,
             (center[0] + d, center[1] - d), (center[0] - d, center[1] + d),
             color, 1, cv2.LINE_AA, 0)


def resample(weights):
    n = len(weights)
    indices = []
    C = [0.] + [sum(weights[:i+1]) for i in range(n)]
    u0, j = np.random.random(), 0
    for u in [(u0+i)/n for i in range(n)]:
        while u > C[j]:
            j += 1
        indices.append(j-1)
    return indices


if __name__ == '__main__':
    # Read the video file
    particle_tracker(video, file_name)
