import rclpy
from rclpy.node import Node
import cv2 as cv
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from vision_msgs.msg import BoundingBox2D
from cv_bridge import CvBridge
import numpy as np


class RedBallTracker(Node):
    def __init__(self):
        super().__init__("Red_Ball_tracker")
        self.subscriber = self.create_subscription(Image,'/camera/image_raw',self.callback,10)
        self.publisher = self.create_publisher(Twist,'/cmd_vel',10)
        self.bridge = CvBridge()
        self.integral_forw = 0
        self.integral_rear = 0
        self.prev_error_forw = 0
        self.prev_error_rear = 0
        self.x_center = 0
        self.y_center = 0
        self.cx = 0
        self.cy = 0
        
    def move(self, contour_area):
        target = 32250
        error = target - contour_area
        dt = 0.1
        kp = 0.000005
        kd = 0.0000001

        derivative = (error - self.prev_error_forw) / dt
        vel = kp * error + kd * derivative

        # clip velocity
        vel = max(-0.15, min(0.15, vel))

        self.prev_error_forw = error

        cmd_vel = Twist()
        cmd_vel.linear.x = vel
        self.publisher.publish(cmd_vel)
        self.get_logger().info(f"Error: {error} \nPublished velocities -> linear: {vel:.2f}")

    def rotate(self,cx):
        error = self.x_center - cx
        self.get_logger().info(f"Rotational error: {error}")
        dt = 0.1
        kp = 0.0002
        ki = 0.00000001
        kd = 0.00001

        # Track integral and derivative terms
        if not hasattr(self, 'integral_rot'):
            self.integral_rot = 0
            self.prev_error_rot = 0

        self.integral_rot += error * dt
        derivative = (error - self.prev_error_rot) / dt
        ang_vel = kp * error + ki * self.integral_rot + kd * derivative
        self.prev_error_rot = error

        # Limit angular velocity
        ang_vel = max(-0.5, min(0.5, ang_vel))

        # Stop if close enough
        if abs(error) < 25:
            ang_vel = 0.0
            self.get_logger().info("Ball centered horizontally. Completed")
            
        cmd_vel = Twist()
        cmd_vel.linear.x = 0.0
        cmd_vel.angular.z = ang_vel 
        self.publisher.publish(cmd_vel)
        self.get_logger().info(f"Published velocities -> linear: {cmd_vel.linear.x:.2f}, angular: {cmd_vel.angular.z:.7f}")

    def callback(self,ros2_img):
        cv_img = self.bridge.imgmsg_to_cv2(ros2_img)
        cv_img_hsv = cv.cvtColor(cv_img,cv.COLOR_RGB2HSV)
        height, width = cv_img.shape[:2]
        self.x_center, self.y_center = width//2, height//2
        cv.circle(cv_img, (self.x_center, self.y_center), 5, (255, 0, 0), -1)
        # lower boundary RED color range values; Hue (0 - 10)
        lower1 = np.array([0, 100, 100])
        upper1 = np.array([10, 255, 255])
        # upper boundary RED color range values; Hue (160 - 180)
        lower2 = np.array([160,110,100])
        upper2 = np.array([179,255,255])
        mask1 = cv.inRange(cv_img_hsv, lower1, upper1)
        mask2 = cv.inRange(cv_img_hsv, lower2, upper2)
        mask = cv.bitwise_or(mask1, mask2)
        kernel = np.ones((8, 8), np.uint8)
        contours,hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
        if contours:
        # get the largest contour by area
            largest_contour = max(contours, key=cv.contourArea)
            max_contour_area = cv.contourArea(largest_contour)
            M = cv.moments(largest_contour)
            if M["m00"] != 0:
                self.cx = int(M["m10"] / M["m00"])
                self.cy = int(M["m01"] / M["m00"])
                cv.circle(cv_img_hsv, (self.cx, self.cy), 5, (0, 0, 0), 2)
            self.get_logger().info(f"Ball detected — Area: {max_contour_area:.2f}")
            cv.circle(cv_img_hsv, (self.x_center, self.y_center), 5, (255, 0, 0), 2) 
            target_area = 32250  # midpoint 
            tolerance = 4000

            if abs(max_contour_area - target_area) > tolerance:
                self.get_logger().info("Ball not at target distance — adjusting position")
                self.move(max_contour_area)
            else:
                self.get_logger().info("Ball within range — rotating for alignment")
                self.rotate(self.cx)
    
            # Draw bounding box
            x, y, w, h = cv.boundingRect(largest_contour)
            cv.rectangle(cv_img_hsv, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        else:
            self.get_logger().info("No Ball detected")

        cv.imshow('Red Ball Tracker', cv.resize(cv_img_hsv, (int(width * 0.5), int(height * 0.5))))
        #cv.imshow('Mask', cv.resize(mask, (750, 500)))
        #cv.imshow('Red Ball Tracker', cv_img_hsv)
        #cv.imshow('Mask', cv.resize(mask, (750, 500)))
        cv.waitKey(25)


def main(args = None):
    rclpy.init(args = args)

    red_ball_tracker = RedBallTracker()
    rclpy.spin(red_ball_tracker)
    red_ball_tracker.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

