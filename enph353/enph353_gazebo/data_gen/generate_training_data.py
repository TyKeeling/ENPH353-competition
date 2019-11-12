#!/usr/bin/env python

# Generates training data for the ML plate reader

# Inputs:
# ./tranining_plates/ : big plate PNG images
# ./backgrounds/ : screen captures from world with no other plates in it. 
# command line: number of images to generate

# Outputs: 
# ./traning_images/XXXX_XX.png : images with big plate images transformed 
# ./traning_images/XXXX_XX.xml : text document with the following information:

# <image>
#   <topleft>
#     <exists>1</exists>
#     <xcoord>14</xcoord>
#     <ycoord>24</ycoord>
#   </topleft>
#   <topright>
#     <exists>1</exists>
#     <xcoord>14</xcoord>
#     <ycoord>24</ycoord>
#   </topright>
#   <bottomleft>
#     <exists>1</exists>
#     <xcoord>14</xcoord>
#     <ycoord>24</ycoord>
#   </bottomleft>
#   <bottomright>
#     <exists>1</exists>
#     <xcoord>14</xcoord>
#     <ycoord>24</ycoord>
#   </bottomright>
# </image>
# 
# Where exists signifies the on page / off page constraint. 

# On the matter of image generation:
# Corners obviously must not be placed randomly. top left must be top right, bottom left must be bottom left
# The deviance should be not too wonky either. We should generate images such that the corners
# are only 30% error from the normal rectangle image. Top left will be randomly placed. 
# Corners may also be placed off screen, which would result in a "0" for "exists" col.

import cv2
import csv
import numpy as np
import os
import pyqrcode
import random
import string
import glob

from random import randint
from PIL import Image, ImageFont, ImageDraw

IMAGE_HEIGHT = 720
IMAGE_WIDTH = 1280

LOOP = 1

# light purple color RGB: 100 100 202   HSV: 240 50.5 79.2
# dark purple color RGB: 0 0 103        HSV: 240 100 40.4


# def find_coeffs(pa, pb): # old points, new points
#     matrix = []
#     for p1, p2 in zip(pa, pb):
#         matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
#         matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])

#     A = np.matrix(matrix, dtype=np.float)
#     B = np.array(pb).reshape(8)

#     res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
#     return np.array(res).reshape(8)

# def shift(img):
#     width, height = img.size

#     coeffs = find_coeffs(
#         [(0, 0), (256, 0), (256, 256), (0, 256)],

#         [(0, 150), (200, 0), (256, 256), (0, 256)])

#     return img.transform((width, height), Image.PERSPECTIVE, coeffs,
#         Image.BICUBIC)

def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")
 
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
 
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
 
	# return the ordered coordinates
	return rect

def four_point_transform(image, pts):
    # SOURCE: Adrian Rosebrock @ https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
 
	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
 
	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
 
	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
 
	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
 
	# return the warped image
	return warped

def main():
    # import backgrounds as a list, increment starting at 0 
    backgrounds = [cv2.imread(file) for file in glob.glob("./backgrounds/*.png")]
    iback = randint(0,len(backgrounds))

    plate_location = [file for file in glob.glob("./training_plates/*.png")]
    plates = [cv2.imread(file) for file in glob.glob("./training_plates/*.png")]
    platelabel = [string.split("/")[2] for string in plate_location]

    pts = [0, 10, 20, 30]
    trans = four_point_transform(backgrounds[0], pts)

    cv2.imshow("Image window",  )
    cv2.waitKey(0)


    for i in range(0, LOOP):
         for plate in plates:


             

             iback = (iback + 1) % len(backgrounds)

if __name__ == '__main__':
    main()