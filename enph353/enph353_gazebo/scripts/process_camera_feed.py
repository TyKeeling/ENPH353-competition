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

        output = purplemask(cv_image, stripes=True)
        image_message = self.bridge.cv2_to_imgmsg(output, encoding="8UC1") #bgr8 or 8UC1
        self.image_out.publish( image_message )


def colormask_contour(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # output = np.zeros(img.shape)
    
                        # Hue Saturation Value(Brightness)
    lower_red = np.array([0,0,90])
    upper_red = np.array([255,50,200])
    
    mask = cv2.inRange(hsv, lower_red, upper_red)

    kernel = np.ones((2,2),np.uint8)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    _, contours, h = cv2.findContours(opening, 1, 2)

    for cnt in contours:
        cv2.drawContours(img,[cnt],0,(0,0,255),-1)
        approx = cv2.approxPolyDP(cnt,0.1*cv2.arcLength(cnt,True),True)
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

    return img

def purplemask(img, stripes=False):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_purple = np.array([120,30,30])
    upper_purple = np.array([130,255,255])
    purplemask = cv2.inRange(hsv, lower_purple, upper_purple)

    kernel = np.ones((2,2),np.uint8)
    opening = cv2.morphologyEx(purplemask, cv2.MORPH_OPEN, kernel)

    #Generate the janky mask around the purple values s.t. we can only find plates. 
    if stripes:
        overmask = np.zeros(purplemask.shape, np.dtype('uint8'))

        for i in range(overmask.shape[0]): #row by row
            for pixel in np.nditer(purplemask[i,1]):
                if pixel != 0:
                    overmask[i] = 255 #make this range white
                    break

        return overmask
    return opening


# Decent idea but this is insanely slow :(
def cluster(img): #https://towardsdatascience.com/introduction-to-image-segmentation-with-k-means-clustering-83fd0a9e2fc3
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    vectorized = hsv.reshape((-1,3))
    vectorized = np.float32(vectorized)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 3
    attempts=10
    ret,label,center=cv2.kmeans(vectorized,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    result_image = res.reshape((img.shape))
    return result_image

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
