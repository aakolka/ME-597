import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist

class PID_control(Node):
    def __init__(self):
        super().__init__('pid_speed_controller')
        self.subscription = self.create_subscription(LaserScan,"/scan",self.dist_receiver,10)    # subscribe to '/scan' 
        self.publisher_ = self.create_publisher(Twist,'/cmd_vel', 10)
       

        self.target = 0.35   # target distance
        self.integral = 0.0
        self.prev_error = 0.0
        self.dt = 0.1

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
        self.forw_dist = msg.ranges[0]
        self.back_dist = msg.ranges[180] 
        self.left_dist = msg.ranges[90] 
        self.right_dist = msg.ranges[270] 
       
    
        self.target = 0.35
        error = self.right_dist - self.target
        derivative = (error - self.prev_error) / self.dt
        self.integral += error * self.dt
        if self.integral > self.integral_max:
            self.integral = self.integral_max
        elif self.integral < self.integral_min:
            self.integral = self.integral_min
        output = self.Kp*error + self.Ki*self.integral + self.Kd*derivative
        
        
        if output > 0.15:
            vel = 0.15
        elif output < -0.15:
            vel = -.15
        else:
            vel = output
        
        
        if self.right_dist - self.target < .01 and self.target - self.right_dist > .01:
            vel = 0
        
        cmd = Twist()
        cmd.linear.x = vel
        self.publisher_.publish(cmd)
        
        #self.get_logger().info(f"\n Current time = {self.elapsed_time:.9f} seconds\n Forward dist={self.forw_dist:.4f} m \n Left dist = {self.left_dist:.4f} m\n Right dist = {self.right_dist:.4f} m \n Back dist = {self.back_dist:.4f} m \n Vel={vel:.4f} m/s\n Error = {error:.4f} m ")
        self.get_logger().info(f"\n n Right dist = {self.right_dist:.4f} m \n Vel={vel:.4f} m/s\n Error = {error:.4f} m ")
        self.prev_error = error
    
    


def main(args=None):
    rclpy.init(args=args)
    controller = PID_control()  # create subscriber node
    rclpy.spin(controller)
    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    controller.destroy_node()
    rclpy.shutdown()
    

if __name__ == '__main__':
    main()
