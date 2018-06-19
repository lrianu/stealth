#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Map homographies using Scale Invariant Feature Transform (SIFT).  Select the
best pairs of matching points via Random Sample Consensus (RANSAC).  Find the
Fundamental Matrix for the homography and estimate its error.

Created on Feb 8 14:49:26 2018

@author: lrianu
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
import warnings


TEXT_X = 1050            # For RMS figure labels
TEXT_Y = 120
units = 'pixels'         # Units used for measures of distance and accuracy
f1 = '/Users/lrianu/Desktop/snapshots/image-088.png'
f2 = '/Users/lrianu/Desktop/snapshots/image-089.png'
RANSAC_THRESHOLD = 3.0


def main():
    # query image (IMG1) and train image (IMG2), grayscale
    img1 = cv2.imread(f1, 0)    # uint8 image, as required by SIFT
    img2 = cv2.imread(f2, 0)    # uint8 image

    # List all keypoints KP1, KP2 in IMG1, IMG2 respectively.
    # List which keypoint matches are GOOD matches
    # Create arrays PTS1, PTS2 with the coordinates of GOOD matches
    kp1, kp2, good, pts1, pts2 = matchKeyPoints(img1, img2)

    # Calculate the homography
    # Draw the homography onto IMG2H
    # Return the mask of best keypoints used to match IMG1 and IMG2
    img2h, maskM = mapHomography(img2, kp1, kp2, good)
    drawMatchingPoints(maskM, img1, img2, kp1, kp2, good)
    F, maskF = findFundamentalMatrix(pts1, pts2) # TODO: verify correct with x2' F x1=0
    showFundamentalMatError(F, maskF, pts1, pts2, img1, img2)


def drawlines(img1, img2, lines1, pts1, pts2):
    '''img1 - image on which we draw the epilines for the points in img2
    lines - corresponding epilines.
    
    Sanity check: these lines should be roughly horizontal'''
    r, c = img1.shape
    imgA = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    imgB = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    for r, pt1, pt2 in zip(lines1, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        imgL = cv2.line(imgA, (x0, y0), (x1, y1), color, 1)
        imgL = cv2.circle(imgA, tuple(pt1), 5, color, -1)
        imgR = cv2.circle(imgB, tuple(pt2), 5, color, -1)
    return imgL, imgR


def fundamentalMatErrorRMS(lines, pts):
    '''Calculate the error of the Fundamental Matrix estimate by finding the
    RMS of the orthogonal distances from points [PTS] to their
    corresponding epipolar LINES.  Epipolar lines can be generated using
    cv2.computeCorrespondEpilines().

    Orthogonal distance formula from point to line:
        d = |ax + by + c| / sqrt(a^2 + b^2)

    lines - [N_LINES x 3] matrix of coefficients for lines of the form
    Ax + By + C = 0.
    pts - [N_PTS x 3] or [N_PTS x 2] matrix of [x y] or homogeneous [x y 1]
    values'''

    # If PTS is not homogeneous, convert to homogenous coordinates.
    if pts.shape[1] == 2:
        pts = np.squeeze(cv2.convertPointsToHomogeneous(pts))
    else:
        pts = np.squeeze(pts)
        
    # TODO: np.matmul?
    dist = np.diagonal(abs(np.dot(lines, np.transpose(pts))))/np.sqrt(
                lines[:, 0]**2 + lines[:, 1]**2)
    rms = np.sqrt((dist**2).sum())
    return rms


def matchKeyPoints(img1, img2):
    '''Match keypoints between IMG1 and IMG2.

    Find keypoints, create keypoint descriptors, nominate putative matches,
    and filter out matches with low confidence.'''

    kp1, kp2, des1, des2 = siftDescriptor(img1, img2)
    matches = flannMatch(des1, des2)
    good, pts1, pts2 = loweRatio(matches, kp1, kp2, LOWE_RATIO=0.7)
    return kp1, kp2, good, pts1, pts2


def siftDescriptor(img1, img2):
    '''Find keypoints and create keypoint descriptors using SIFT'''
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # Find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    return kp1, kp2, des1, des2


def flannMatch(des1, des2):
    '''Try to match keypoint descriptors using FLANN (Fast Library for
    Approximate Nearest Neighbors) KDTREE method'''
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)
    return matches


def loweRatio(matches, kp1, kp2, LOWE_RATIO=0.7):
    """Reduce list of keypoints to likeliest matches using Lowe's Ratio test.

    [Reference]
    David G. Lowe. Distinctive Image Features from Scale-Invariant Keypoints.
    IJCV 2004"""
    # Store all the good matches as per Lowe's ratio test
    good = []
    pts1 = []
    pts2 = []

    for m, n in matches:
        if m.distance < LOWE_RATIO * n.distance:
            good.append(m)
            pts1.append(kp1[m.queryIdx].pt)
            pts2.append(kp2[m.trainIdx].pt)
    return good, pts1, pts2


def mapHomography(img, kp1, kp2, good, MIN_MATCH_COUNT=10):
    '''Calculate the homography between GOOD keypoint matches and map onto
    image IMGH'''
    # If at least MIN_MATCH_COUNT matches are there for mapping, continue.
    if len(good) > MIN_MATCH_COUNT:
        M, matchesMask = homography(kp1, kp2, good)
        imgH = showTransform(img, M)
    else:
        print("Not enough matches found - {}/{}".format(
                len(good), MIN_MATCH_COUNT))
        matchesMask = None
    return imgH, matchesMask


def homography(kp1, kp2, good):
    '''Find the homography between GOOD matches in KP1 and KP2.

    ransacReprojThreshold - [int] -  RANSAC reprojection threshold'''
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,
                                 RANSAC_THRESHOLD)
    matchesMask = mask.ravel().tolist()
    return M, matchesMask


def showTransform(img, M):
    '''Draw the transform M onto IMG'''
    h, w = img.shape
    pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(
                -1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)
    imgH = cv2.polylines(img, [np.int32(dst)], isClosed=True,
                         color=255, thickness=3, lineType=cv2.LINE_AA)
    return imgH


def findFundamentalMatrix(pts1, pts2):
    # Find the Fundamental Matrix
#    F, mask = cv2.findFundamentalMat(np.int32(pts1), np.int32(pts2),
#                                     cv2.FM_RANSAC, RANSAC_THRESHOLD)
    F, mask = cv2.findFundamentalMat(np.int32(pts1), np.int32(pts2),
                                     cv2.FM_RANSAC)
    # TODO: Learn heuristics for tuning the reprojection threshold.
    mask = [int(i) for i in np.squeeze(mask)]

    # Check the rank of the Fundamental Matrix
    rank = np.linalg.matrix_rank(F)
    if rank > 2:
        # TODO: correct for rank 3 Fundamental matrices
        warnings.warn('Fundamental matrix with rank 3')
    return F, mask


def drawMatchingPoints(matchesMask, img1, img2, kp1, kp2, good):
    '''Plot side by side query and train images with matching points
    connected'''
    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,   # draw only inliers
                       flags=2)
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
    fig = plt.figure(figsize=(18, 16), dpi=80, facecolor='w', edgecolor='k')
    plt.imshow(img3, 'gray', aspect='equal'), plt.show()
    return None


def showFundamentalMatError(F, mask, pts1, pts2, img1, img2):
    lines1, lines2, pts1, pts2 = calculateEpilines(F, mask, pts1, pts2)
    img4, img5 = drawlines(img2, img1, lines2, pts2, pts1)
    img6, img7 = drawlines(img1, img2, lines1, pts1, pts2)

    # Calculate homography error as RMS between points and epipolar lines
    rms1 = fundamentalMatErrorRMS(lines1, pts1)
    rms2 = fundamentalMatErrorRMS(lines2, pts2)

    # Show epipolar lines on both images
    fig = plt.figure(figsize=(18, 16), dpi=80, facecolor='w', edgecolor='k')
    plt.subplot(121), plt.imshow(img6)
    plt.text(TEXT_X, TEXT_Y,
             'Fundamental\nMatrix Error\n(RMS) = '
             '{:4.1f}\n{}'.format(rms1, units),
             color='w')
    plt.subplot(122), plt.imshow(img4)
    plt.text(TEXT_X, TEXT_Y,
             'Fundamental\nMatrix Error\n(RMS) = '
             '{:4.1f}\n{}'.format(rms2, units),
             color='w')
    plt.show(fig)
    return None


def calculateEpilines(F, mask, pts1, pts2):
    '''Calculate and display epilines as projected from left image onto right,
    and from right image onto left.'''
    # Use only inliers
    tf = (np.squeeze(mask) == 1)

    ptsL = np.float32(pts1)[tf]
    ptsR = np.float32(pts2)[tf]

    # Find epilines corresponding to points in right [left] image and
    # drawing its lines on left [right] image
    linesL = cv2.computeCorrespondEpilines(ptsR.reshape(-1, 1, 2), 2, F)
    linesR = cv2.computeCorrespondEpilines(ptsL.reshape(-1, 1, 2), 1, F)
    linesL = linesL.reshape(-1, 3)
    linesR = linesR.reshape(-1, 3)
    return linesL, linesR, ptsL, ptsR


if __name__ == '__main__':
    main()
