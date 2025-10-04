import rclpy
from rclpy.node import Node

from task_2_interfaces.msg import JointData


class MinimalPublisher(Node):
    def __init__(self):
        super().__init__("talker")
        self.publisher_ = self.create_publisher(JointData, 'joint_topic', 10)
        timer_period = 1
        self.timer = self.create_timer(timer_period,self.timer_callback)
        self.i = 0.0

    def timer_callback(self):
        msg = JointData()
        msg.vel = self.i
        self.publisher_.publish(msg)
        self.i += 1.0

def main(args=None):
    rclpy.init(args=args)

    minimal_publisher = MinimalPublisher()

    rclpy.spin(minimal_publisher)

    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()



