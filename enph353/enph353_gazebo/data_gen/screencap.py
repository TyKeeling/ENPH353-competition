#!/usr/bin/env python
from __future__ import division

import sys
import time

import numpy as np
import os, os.path

import cv2
import roslib
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import Int16, String

expected_error_max = 100

Kernel_size = 15
low_threshold = 75
high_threshold = 110
bwThresh = 100


class image_converter:

    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(
            "/R1/pi_camera/image_raw", Image, self.callback)
        self.i = len(os.listdir('./backgrounds')) + 1

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        if (self.i % 5 == 0):
            cv2.imwrite('backgrounds/img{0:03d}.png'.format(self.i), cv_image)
        self.i = self.i+1


def main(args):
    rospy.init_node('image_converter', anonymous=True)
    ic = image_converter()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)
