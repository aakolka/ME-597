
# ME 597 - Autonomous Systems

This repository contains code written for ME 597: Autonomous Systems at Purdue University. There are a total of 6 tasks in 4 labs, covering algorithms used for path planning, navigation and perception, along with a final project. All code is written in Python.

---
A short description of every lab task is given below:

**Lab 1**

- Task 1: Write a ROS2 python package with a publisher, subscriber and launch script
- Task 2: Write a ROS2 python package with:
    - A simple publisher and subscriber node using a custom msg
    - A simple service server and service client node using a custom srv

**Lab 2**
- Task 3: PID distance controller using LiDAR data to allow a TurtleBot4 to stop 'x' meters from an obstacle directly in front of it.

**Lab 3**
- Task 4: Autonomous path planning and following in a known environment.

**Lab 4**
- Task 5: Publish video frames and detect a red triangle using ROS 2, outputting its position, size, and bounding box.
- Task 6: Track a red ball in simulation with ROS 2, controlling the robot via PID to follow and center the ball.

**Final Project**

The final project was implementing a SLAM algorithm to autonomously explore and map an unknown environment. 



## Installation

1. Clone the required simulation workspace (```sim_ws```)

```bash
  git clone https://github.com/Purdue-ME597/sim_ws
```
2. In a new terminal build the sim_ws workspace:
```
cd sim_ws
colcon build --symlink-install
```

3. Add turtlebot3 environment variable to .bashrc file
```
echo "export TURTLEBOT3_MODEL=waffle" >> ~/.bashrc
```

4. Run these to install packages to be used in the simulator.
```
sudo apt install ros-humble-turtlebot3-teleop
sudo apt install ros-humble-slam-toolbox
sudo apt install ros-humble-navigation2
```
```
pip install pynput
```
5. Install OpenCV and cv_bridge in ROS2
```
sudo apt install python3-numpy
sudo apt install libboost-python-dev
sudo apt install python3-opencv
```
6. Clone the OpenCV repo
```
cd <YOUR_ROS2_WORKSPACE>/src
git clone https://github.com/ros-perception/vision_opencv.git -b rolling
cd ..
colcon build --symlink-install
```
7. Install the repo
```
git clone https://github.com/aakolka/ME-597.git
```
