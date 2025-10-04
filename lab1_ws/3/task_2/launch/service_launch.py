from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    x_arg = DeclareLaunchArgument('x', default_value='0.0') # x,y,z position argument
    y_arg = DeclareLaunchArgument('y', default_value='0.0')
    z_arg = DeclareLaunchArgument('z', default_value='0.0')
    return LaunchDescription([   # Return the launch description containing all actions
        x_arg,
        y_arg,
        z_arg,
        Node(
            package='task_2',
            executable='service',
            name='Service_Server'   # service node
        ),
        
        Node(
            package='task_2',
            executable='talker',
            name='talker'   # Node name for the talker
        ),
        Node(
            package='task_2',
            executable='listener',
            name='listener'  # Node name for the listener
        ),
            ]
        )



