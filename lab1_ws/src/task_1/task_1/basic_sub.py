import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32

class BasicSubscriber(Node):
    
    def __init__(self):
        super().__init__("listener")
        self.subscription = self.create_subscription(Int32,"my_first_topic",self.listener_callback,10)    # subscribe to 'my_first_topic' 

    def listener_callback(self, msg):
        # log the message
        self.get_logger().info('Publishing: This node has been active for %d seconds, half of %d seconds' % (msg.data, 2*msg.data))

def main(args=None):
    rclpy.init(args=args)
    minimal_subscriber = BasicSubscriber()  # create subscriber node
    rclpy.spin(minimal_subscriber)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
