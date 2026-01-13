import rclpy  # ROS2 client library
from rclpy.node import Node
from std_msgs.msg import Int32  # import message type Int32

class BasicPublisher(Node):

    def __init__(self):
        super().__init__("talker")   # initialize node
        self.publisher_ = self.create_publisher(Int32, 'my_first_topic', 10)   # create publisher
        timer_period = 1
        self.timer = self.create_timer(timer_period,self.timer_callback)  # call timer_callback every second
        self.i = 0

    def timer_callback(self):
        msg = Int32()
        msg.data = self.i    # set message data
        self.publisher_.publish(msg)  # Publish message
        self.i += 1

def main(args=None):
    rclpy.init(args=args)   # initialize ROSs
    minimal_publisher = BasicPublisher()  # create publisher node
    rclpy.spin(minimal_publisher)
    minimal_publisher.destroy_node()  # Destroy the node explicitly
    rclpy.shutdown()

if __name__ == '__main__':
    main()


