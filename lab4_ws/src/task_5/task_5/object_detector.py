"""
Author: Aryaman Akolkar
Course: ME 597, Purdue University
Semester: Fall 2025

Task: 5B
Goal: Real-time object detection and tracking of a red triangular object from a 
video stream using ROS 2 and OpenCV
"""

import rclpy
from rclpy.node import Node
import cv2 as cv
from sensor_msgs.msg import Image
from vision_msgs.msg import BoundingBox2D
from cv_bridge import CvBridge
import numpy as np

class ObjectDetector(Node):
    def __init__(self):
        """
          Args:
            - None
        Returns:
            - None
        """
        super().__init__("image_subscriber")
        self.subscriber = self.create_subscription(Image,'/video_data',self.callback,10)
        self.info_publisher = self.create_publisher(BoundingBox2D,'/bbox',10)
        
        # Bridge for ROS and OpenCV image conversion
        self.bridge = CvBridge()
    
    def callback(self,ros2_img):
        """
        Callback function that processes incoming image frames,
        detects the red triangular object, and publishes its bounding box.
        Args:
            - ros2_img : ROS Image message
        Returns:
            - None
        """
        cv_img = self.bridge.imgmsg_to_cv2(ros2_img)
        cv_img_hsv = cv.GaussianBlur(cv.cvtColor(cv_img,cv.COLOR_BGR2HSV),(3,3),0)
        
        # lower boundary RED color range values; Hue (0 - 10)
        lower1 = np.array([0, 110, 190])
        upper1 = np.array([5, 255, 255])
        
        # upper boundary RED color range values; Hue (160 - 180)
        lower2 = np.array([175,110,190])
        upper2 = np.array([179,255,255])

        # Create binary masks for red color ranges
        mask1 = cv.inRange(cv_img_hsv, lower1, upper1)
        mask2 = cv.inRange(cv_img_hsv, lower2, upper2)
        mask = cv.bitwise_or(mask1, mask2)

        kernel = np.ones((8, 8), np.uint8)
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

        # Find contours
        contours,hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        max_contour_area = 0
        index = -1

        # Select the largest contour
        if len(contours) > 0:
            for i in range(len(contours)):
                area = cv.contourArea(contours[i])
                if area > max_contour_area:
                    max_contour_area = area
                    index = i
        if index != -1:
            # Compute centroid
            M = cv.moments(contours[index])
            if M['m00'] != 0:
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])

            x,y,w,h = cv.boundingRect(contours[index])
            cv.rectangle(cv_img,(x,y),(x+w,y+h),(0,255,0),2)
            msg = BoundingBox2D()
            msg.size_x = float(w)
            msg.size_y = float(h)
            msg.center.position.x = float(cx)
            msg.center.position.y = float(cy)
            self.info_publisher.publish(msg)

        cv.imshow('Video',cv_img)
        cv.imshow('Mask',mask)
        cv.waitKey(20)

def main(args = None):
    rclpy.init(args = args)
    object_detector = ObjectDetector()
    rclpy.spin(object_detector)
    object_detector.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

