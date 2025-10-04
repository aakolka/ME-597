from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='task_1',
            executable='talker',
            name='talker'  # publisher node
        ),
        Node(
            package='task_1',
            executable='listener',
            name='listener' # subscriber node
        ),
            ]
        )
    
