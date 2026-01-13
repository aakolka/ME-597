import launch
import launch_ros.actions


def generate_launch_description():
    return launch.LaunchDescription([
        launch_ros.actions.Node(
            package='task_1',
            executable='talker',
            name='Talker_node'),
        launch_ros.actions.Node(
            package='task_1',
            executable='listener',
            name='Listener_node'),
    ])