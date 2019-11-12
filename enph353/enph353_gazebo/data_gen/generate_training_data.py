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

from random import randint
from PIL import Image, ImageFont, ImageDraw

IMAGE_HEIGHT = 720
IMAGE_WIDTH = 1280

# light purple color RGB: 100 100 202   HSV: 240 50.5 79.2
# dark purple color RGB: 0 0 103        HSV: 240 100 40.4
def main(args):
    # import backgrounds as a list, increment starting at 0 

    for i in range(0,args[1]):


        for plate in traning_plates:
        
if __name__ == '__main__':
    main(sys.argv)