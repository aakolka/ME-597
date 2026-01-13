from task_2_interfaces.srv import JointState       # import custom service message type                                                    
import rclpy
from rclpy.node import Node


class Task2_Service(Node):

    def __init__(self):
        super().__init__('Task2_service')   #Initialize node
        self.srv = self.create_service(JointState, 'joint_service', self.handle_request)       # Create service node

    def handle_request(self, request, response):
        total = request.x + request.y + request.z   #Sum of requested fields                                                
        response.valid = total >= 0.0               
        self.get_logger().info(f'Received request: x={request.x}, y={request.y}, z={request.z} | sum={total} -> valid={response.valid}')  #log request

        return response

def main(args=None):
    rclpy.init(args=args)  #Initialize ROS

    task2_service = Task2_Service()

    rclpy.spin(task2_service)  #Create node

    rclpy.shutdown()

if __name__ == '__main__':
    main()