"""
Author: Aryaman Akolkar
Course: ME 597, Purdue University
Semester: Fall 2025

Task: 5A
Goal: Publish video frames as ROS2 Image messages to simulate a live camera feed 
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import numpy as np
import cv2 as cv 
from cv_bridge import CvBridge
import time

class ImagePublisher(Node):
    def __init__(self):
        self.bridge = CvBridge()
        super().__init__("image_publisher")
        self.publisher = self.create_publisher(Image,'/video_data',10)

    def frame_cap(self):
        """
        Captures frames from a video file and publishes them at a fixed rate to
        emulate a real-time camera stream. The video loops continuously once
        the end is reached.
        """
        self.cap = cv.VideoCapture('/home/me597/lab4_ws/src/task_5/resource/lab3_video.avi')
        self.i = 0
        if not self.cap.isOpened():
            print("Cannot play video")
            return
    
        fps = 25  # desired playback speed (adjust if needed)
        frame_delay = 1.0 / fps

        while self.cap.isOpened():
            ret,frame = self.cap.read()
            
            # Restart video from the beginning
            if not ret:
                #break
                self.cap.set(cv.CAP_PROP_POS_FRAMES, 0)
                continue
            
            # Convert OpenCV frame to ROS2 Image
            ros2_img = self.bridge.cv2_to_imgmsg(frame)
            self.publisher.publish(ros2_img)
            print(f'Frame #{self.i} published.')
            self.i += 1   
            time.sleep(frame_delay)
            
            if cv.waitKey(30) & 0xFF == ord('q'):
                break
        self.cap.release()

def main(args=None):
    rclpy.init(args=args)   # initialize ROS
    image_publisher = ImagePublisher()  # create publisher node
    image_publisher.frame_cap() 
    rclpy.spin(image_publisher)
    image_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()






