from task_2_interfaces.srv import JointState   #Import custom message type
import sys
import rclpy
from rclpy.node import Node


class MinimalClientAsync(Node):

    def __init__(self):
        super().__init__('task2_client')   #Initialize node
        self.cli = self.create_client(JointState, 'joint_service')       #Create client
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = JointState.Request()    #create request object

    def send_request(self):
        self.req.x = float(sys.argv[1])   #set x,y and z values
        self.req.y = float(sys.argv[2])
        self.req.z = float(sys.argv[3])    
        self.future = self.cli.call_async(self.req)  # send request


def main(args=None):
    rclpy.init(args=args)  # initialize ROS

    task2_client = MinimalClientAsync()
    task2_client.send_request()

    while rclpy.ok():
        rclpy.spin_once(task2_client)
        if task2_client.future.done():   # check if response received
            try:
                response = task2_client.future.result()
            except Exception as e:
                task2_client.get_logger().info(
                    'Service call failed %r' % (e,))
            else:
                task2_client.get_logger().info(  # log response
                    f'Result: {response.valid} (inputs: {task2_client.req.x}, {task2_client.req.y}, {task2_client.req.z})')
            break

    task2_client.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()