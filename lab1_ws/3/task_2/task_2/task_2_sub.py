import rclpy
from rclpy.node import Node
from task_2_interfaces.msg import JointData

class BasicSubscriber(Node):
    def __init__(self):
        super().__init__("listener")
        self.subscription = self.create_subscription(JointData,"joint_topic",self.listener_callback,10)

    def listener_callback(self, msg):
        self.get_logger().info('Publishing: This node has been active for %d seconds, half of %d seconds' % (msg.vel, 2*msg.vel))

def main(args=None):
    rclpy.init(args=args)
    minimal_subscriber = BasicSubscriber()
    rclpy.spin(minimal_subscriber)
    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
