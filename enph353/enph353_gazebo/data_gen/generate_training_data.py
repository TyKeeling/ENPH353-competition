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
#   <squarer>10</squarer>
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
import math

from random import randint
from PIL import Image, ImageFont, ImageDraw
import xml.etree.cElementTree as ET
from xml.dom import minidom

# Thank you to Matthew Earl https://github.com/matthewearl for the use of the below functions from 
# https://github.com/matthewearl/deep-anpr/blob/master/gen.py , specifically the perspective transforms.
# Copyright (c) 2016 Matthew Earl

def euler_to_mat(yaw, pitch, roll):
    # Rotate clockwise about the Y-axis
    c, s = math.cos(yaw), math.sin(yaw)
    M = np.matrix([[  c, 0.,  s],
                      [ 0., 1., 0.],
                      [ -s, 0.,  c]])

    # Rotate clockwise about the X-axis
    c, s = math.cos(pitch), math.sin(pitch)
    M = np.matrix([[ 1., 0., 0.],
                      [ 0.,  c, -s],
                      [ 0.,  s,  c]]) * M

    # Rotate clockwise about the Z-axis
    c, s = math.cos(roll), math.sin(roll)
    M = np.matrix([[  c, -s, 0.],
                      [  s,  c, 0.],
                      [ 0., 0., 1.]]) * M

    return M

def make_affine_transform(from_shape, to_shape, 
                          min_scale, max_scale,
                          scale_variation=1.0,
                          rotation_variation=1.0,
                          translation_variation=1.0):
    out_of_bounds = False

    from_size = np.array([[from_shape[1], from_shape[0]]]).T
    to_size = np.array([[to_shape[1], to_shape[0]]]).T

    scale = random.uniform((min_scale + max_scale) * 0.5 -
                           (max_scale - min_scale) * 0.5 * scale_variation,
                           (min_scale + max_scale) * 0.5 +
                           (max_scale - min_scale) * 0.5 * scale_variation)
    if scale > max_scale or scale < min_scale:
        out_of_bounds = True
    roll = random.uniform(-0.3, 0.3) * rotation_variation
    pitch = random.uniform(-0.2, 0.2) * rotation_variation
    yaw = random.uniform(-1.2, 1.2) * rotation_variation

    # Compute a bounding box on the skewed input image (`from_shape`).
    M = euler_to_mat(yaw, pitch, roll)[:2, :2]
    h, w = from_shape
    corners = np.matrix([[-w, +w, -w, +w],
                            [-h, -h, +h, +h]]) * 0.5
    skewed_size = np.array(np.max(M * corners, axis=1) -
                              np.min(M * corners, axis=1))

    # Set the scale as large as possible such that the skewed and scaled shape
    # is less than or equal to the desired ratio in either dimension.
    scale *= np.min(to_size / skewed_size)

    # Set the translation such that the skewed and scaled image falls within
    # the output shape's bounds.
    trans = (np.random.random((2,1)) - 0.5) * translation_variation
    trans = ((2.0 * trans) ** 5.0) / 2.0
    if np.any(trans < -0.5) or np.any(trans > 0.5):
        out_of_bounds = True
    trans = (to_size - skewed_size * scale) * trans

    center_to = to_size / 2.
    center_from = from_size / 2.

    M = euler_to_mat(yaw, pitch, roll)[:2, :2]
    M *= scale
    M = np.hstack([M, trans + center_to - M * center_from])

    return M, out_of_bounds, trans

def prettify(elem): # https://pymotw.com/2/xml/etree/ElementTree/create.html
    rough_string = ET.ElementTree.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

def make_coord_xml(coords):   # https://stackoverflow.com/questions/3605680/creating-a-simple-xml-file-using-python
    # make sure labels follows the same ordering convention as the inputs
    labels = ["topleft", "topright", "bottomleft", "bottomright"]
    image = ET.Element("image")

    # child = SubElement(top, 'child')
    # child.text = 'This child contains text.'
    i = 0
    for label in labels:
        coord = coords[i]

        a = ET.SubElement(image, label)
        exists = ET.SubElement(a, "exists")
        xcoord = ET.SubElement(a, "xcoord")
        ycoord = ET.SubElement(a, "ycoord")

        if (coord[0] == 0 and coord[1] == 0):
            exists.text = "0"
        else:
            exists.text = "1"

        xcoord.text = str(coord[0])
        ycoord.text = str(coord[1])

        i += 1 

    # https://codeblogmoney.com/xml-pretty-print-using-python-with-examples/
    return ET.ElementTree(image)

def createimages(backgrounds, plates):
    outimage = []
    coordtree = []

    iback = randint(0,len(backgrounds))-1

    for plate in plates:
        bg = backgrounds[iback]
        plate = cv2.resize(plate, (plate.shape[0]/2, plate.shape[1])) # scaling down by 2
        plate = plate * random.uniform(0.5,0.9) #shading
        # light purple color RGB: 100 100 202   HSV: 240 50.5 79.2
        # dark purple color RGB: 0 0 103        HSV: 240 100 40.4
        prand = random.random()
        purple = [240, int(40+70*prand), int(90-60*prand)] 

        bordersize = 200
        plate = cv2.copyMakeBorder(
            plate,
            top=0,
            bottom=0,
            left=bordersize,
            right=bordersize,
            borderType=cv2.BORDER_CONSTANT,
            value=purple
        )

        plate = plate.astype(np.uint8)
        platesize = np.ones(plate.shape, np.uint8)
        platesize = platesize * 255 #white mask
        cornerplate = np.zeros(plate.shape, np.uint8) # blank for adding corners.
        cornersize = 25 # size of contour

        (bottom, right, _) = cornerplate.shape
        topleft = cv2.circle(cornerplate.copy(), (0+bordersize,    0),      1, (255,255,255), cornersize) 
        topright = cv2.circle(cornerplate.copy(), (right-bordersize,0),      1, (255,255,255), cornersize)
        bottomleft = cv2.circle(cornerplate.copy(), (0+bordersize,    bottom), 1, (255,255,255), cornersize) 
        bottomright = cv2.circle(cornerplate.copy(), (right-bordersize,bottom), 1, (255,255,255), cornersize)
        cornerimg = [topleft, topright, bottomleft, bottomright]

        # Transforming made plates 
        M, out_of_bounds, trans = make_affine_transform(
                from_shape=(plate.shape[0], plate.shape[1]),
                to_shape=(bg.shape[0], bg.shape[1]),
                min_scale=0.3,
                max_scale=0.4,
                rotation_variation=1.0,
                scale_variation=3.5,
                translation_variation=1.2)
        #center = (trans[0]+bg.shape[1]/2, trans[1]+bg.shape[0]/2)

        # https://pythonprogramming.net/color-filter-python-opencv-tutorial/
        plate_T = cv2.warpAffine(plate, M, (bg.shape[1], bg.shape[0]))
        mask = cv2.warpAffine(platesize, M, (bg.shape[1], bg.shape[0]))
        out = cv2.bitwise_and(bg, cv2.bitwise_not(mask)) + cv2.bitwise_and(plate_T, mask)

        cornercont = [] # Should be list of only 4
        for c in cornerimg:
            c = cv2.warpAffine(c, M, (bg.shape[1], bg.shape[0]))
            c = cv2.cvtColor(c, cv2.COLOR_RGB2GRAY)
            _, contours, _ = cv2.findContours(c, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cornercont.append(contours)

        #Get coordinate points
        coords = [(0,0)] * 4
        #squarer = 10

        for i in range(len(cornercont)): #4
            for c in cornercont[i]:
                # compute the center of the contour
                M = cv2.moments(c)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                #cv2.rectangle(out, (cX-squarer,cY-squarer), (cX+squarer,cY+squarer), (0,255,0), 1 )
                coords[i] = (cX,cY) 

        outimage.append(out)
        coordtree.append(make_coord_xml(coords))
        iback = (iback + 1) % len(backgrounds)
    return outimage, coordtree 

def main():
    # coords = [ (0,2), (0,0), (1,5), (10,20)]
    # xml = make_coord_xml(coords)
    # xml.write("output.xml")

    backgrounds = [cv2.imread(file) for file in glob.glob("./backgrounds/*.png")]
    plates = [cv2.imread(file) for file in glob.glob("./training_plates/*.png")]
    plate_location = [file for file in glob.glob("./training_plates/*.png")]
    platelabel = [string.split("/")[2] for string in plate_location]

    plateloop = 30
    for j in range(plateloop):
        outimages, coords = createimages(backgrounds, plates)

        for i in range(len(outimages)):
            cv2.imwrite("training_images/"+"{:03d}_".format(i+j)+platelabel[i],
                outimages[i])

            coords[i].write("training_images/" +
                "{:03d}_".format(i+j) + 
                platelabel[i].split(".")[0] + 
                ".xml")

if __name__ == '__main__':
    main()