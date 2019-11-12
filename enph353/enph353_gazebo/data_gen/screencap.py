#!/usr/bin/env python
from __future__ import division

import sys
import time

import numpy as np

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
        self.image_out = rospy.Publisher("/R1/image_out", Image, queue_size=10)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(
            "/R1/pi_camera/image_raw", Image, self.callback)

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        output = process_image(cv_image)
        image_message = self.bridge.cv2_to_imgmsg(output, encoding="bgr8")
        self.image_out.publish( image_message )


def process_image(img):

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
                        # Hue Saturation Value(Brightness)
    lower_red = np.array([0,0,90])
    upper_red = np.array([255,10,200])
    
    mask = cv2.inRange(hsv, lower_red, upper_red)

    kernel = np.ones((2,2),np.uint8)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


    image, contours, h = cv2.findContours(opening,1,2)

    for cnt in contours:
        approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
        #print len(approx)
        # if len(approx)==5:
        #     print "pentagon"
        #     cv2.drawContours(img,[cnt],0,255,-1)
        # elif len(approx)==3:
        #     print "triangle"
        #     cv2.drawContours(img,[cnt],0,(0,255,0),-1)

        if len(approx)==4:
            x,y,w,h = cv2.boundingRect(cnt)
            if w > 50 and h > 50 and cv2.contourArea(cnt) > 500:
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
                #print cv2.moments(cnt)
                cv2.drawContours(img,[cnt],0,(0,0,255),-1)

    # image, contours, hierarchy = cv2.findContours(opening,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    # output = cv2.drawContours(image, contours, -1, (0,255,0), 3)
    
    #res = cv2.bitwise_and(img, img, mask= opening)

    #cv2.imshow("Image window", edged)
    #cv2.waitKey(0)

    return img


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
