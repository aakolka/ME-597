from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    '''
    Launches the pid_controller node from task_3 package
    '''
    return LaunchDescription([
        Node(
            package='task_3',
            executable='pid_speed_controller',
        ),
            ]
        )
