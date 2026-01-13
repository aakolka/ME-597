"""
Author: Aryaman Akolkar
Course: ME 597, Purdue University
Semester: Fall 2025

Task: 3
Goal: PID distance controller using lidar data to allow the TurtleBot4 to 
stop 'x' meters from an obstacle directly in front of it.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist

class PID_control(Node):
    def __init__(self):
        '''
        Initializes the PID_control class
        Args: 
            -None
        Returns:
            -None
        '''
        super().__init__('pid_speed_controller')
        self.subscription = self.create_subscription(LaserScan,"/scan",self.dist_receiver,10)    # subscribe to '/scan' 
        self.publisher_ = self.create_publisher(Twist,'/cmd_vel', 10)
       
        self.target = 0.35   # target distance to wall
        self.integral = 0.0
        self.prev_error = 0.0
        self.dt = 0.1

        #PID gains
        self.Kp = .5
        self.Ki = .05
        self.Kd = .1

        self.integral_min = -1.0
        self.integral_max = 1.0
        self.forw_dist = 0
        self.back_dist = 0
        self.left_dist = 0
        self.right_dist = 0
        
    def dist_receiver(self, msg):
        '''
        Processes LiDAR data and applies PID control to compute and publish a forward velocity command.
        Args:
            - msg (LaserScan): Incoming LiDAR scan message.
        Returns:
            - None
        '''
        self.forw_dist = msg.ranges[0]
        self.back_dist = msg.ranges[180] 
        self.left_dist = msg.ranges[90] 
        self.right_dist = msg.ranges[270] #Provides forward distance
       
        # PID controller
        self.target = 0.35
        error = self.right_dist - self.target
        derivative = (error - self.prev_error) / self.dt
        self.integral += error * self.dt

        # Anti windup
        if self.integral > self.integral_max:
            self.integral = self.integral_max
        elif self.integral < self.integral_min:
            self.integral = self.integral_min
        output = self.Kp*error + self.Ki*self.integral + self.Kd*derivative
        
        # Velocity saturation
        if output > 0.15:
            vel = 0.15
        elif output < -0.15:
            vel = -.15
        else:
            vel = output
        
        # Near-target deadband
        if self.right_dist - self.target < .01 and self.target - self.right_dist > .01:
            vel = 0
        
        # Publish velocity command
        cmd = Twist()
        cmd.linear.x = vel
        self.publisher_.publish(cmd)
        
        self.get_logger().info(f"\n n Dist = {self.right_dist:.4f} m \n Vel={vel:.4f} m/s\n Error = {error:.4f} m ")
        self.prev_error = error

def main(args=None):
    rclpy.init(args=args)
    controller = PID_control()  # create subscriber node
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()
    
if __name__ == '__main__':
    main()
