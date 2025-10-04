import os

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    slam_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                '/opt/ros/humble/share/turtlebot4_navigation/launch',
                'slam.launch.py'
            )
        )
    )

    map_view = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                '/opt/ros/humble/share/turtlebot4_viz/launch',
                'view_robot.launch.py'
            )
        )
    )

    return LaunchDescription([
        slam_launch,
        map_view
    ])
