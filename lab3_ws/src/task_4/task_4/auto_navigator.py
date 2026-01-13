"""
Author: Aryaman Akolkar
Course: ME 597, Purdue University
Semester: Fall 2025

Task: 4
Goal: Autonomous path planning and navigation in a known environment

NOTE:
All classes in this file, except for the Navigation class, were provided by the course instructors. 
The Map, MapProcessor, Queue, Node, Tree, and AStar classes are instructor-supplied; 
however, the solve function within the AStar class, along with the entire Navigation class, 
was fully implemented by the author to enable autonomous path planning and path following
"""

#!/usr/bin/env python3
from PIL import Image, ImageOps
from ament_index_python.packages import get_package_share_directory
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import yaml
from copy import copy, deepcopy
import math
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Pose, Twist
from std_msgs.msg import Float32

class Navigation(Node):
    def __init__(self, map_processor,node_name='Navigation'):
        """
        Initializes the Navigation node for path planning and following.
        Args:
            - map_processor : MapProcessor instance containing the map and graph
            - node_name : ROS2 node name (default 'Navigation')
        Returns:
            - None
        """
        super().__init__(node_name)
        self.map_processor = map_processor
        self.goal_pose = PoseStamped()
        self.ttbot_pose = PoseStamped()
        self.start_time = 0.0
        self.last_path_idx = 0
        self.resolution = float(self.map_processor.map.map_df.resolution[0])
        self.origin = self.map_processor.map.map_df.origin[0]
        self.x_min = float(self.origin[0])
        self.y_min = float(self.origin[1])
        self.map_height = self.map_processor.map.image_array.shape[0]
        self.goal_reached = False
        self.replan_needed = False

        # Subscribers
        self.create_subscription(PoseStamped, '/move_base_simple/goal', self.__goal_pose_cbk, 10)
        self.create_subscription(PoseWithCovarianceStamped, '/robot/amcl_pose', self.__ttbot_pose_cbk, 10)

        # Publishers
        self.path_pub = self.create_publisher(Path, 'global_plan', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/robot/cmd_vel', 10)
        self.calc_time_pub = self.create_publisher(Float32, 'astar_time',10) #DO NOT MODIFY

        # Node rate
        self.rate = self.create_rate(10)

    def __goal_pose_cbk(self, data):
        """
        ROS callback for receiving the goal from RViz.
        Args:
            - data : PoseStamped from RViz
        Returns:
            - None
        """
        self.goal_pose = data
        self.get_logger().info(
            'goal_pose: {:.4f}, {:.4f}'.format(self.goal_pose.pose.position.x, self.goal_pose.pose.position.y))
    
    def __ttbot_pose_cbk(self, data):
        """
        ROS callback for receiving the robot pose from AMCL.
        Args:
            - data : PoseWithCovarianceStamped
        Returns:
            - None
        """
        self.ttbot_pose = data.pose
        self.get_logger().info(
            'ttbot_pose: {:.4f}, {:.4f}'.format(self.ttbot_pose.pose.position.x, self.ttbot_pose.pose.position.y))

    def pose_to_grid(self, vehicle_pose):
        """
        Converts world coordinates to grid indices in the map.
        Args:
            - vehicle_pose : PoseStamped object
        Returns:
            - i, j : grid indices corresponding to the vehicle position
        """
        x_world = vehicle_pose.pose.position.x
        y_world = vehicle_pose.pose.position.y
        j = int((x_world - self.x_min) / self.resolution)  
        i = int(self.map_height - 1 - (y_world - self.y_min) / self.resolution)
    
        return i, j  
    
    def grid_to_pose(self,i, j):
        """
        Converts grid indices to world coordinates.
        Args:
            - i : row index
            - j : column index
        Returns:
            - x_world, y_world : corresponding world coordinates
        """
        x_world = self.x_min + (j + 0.5) * self.resolution
        y_world = self.y_min + (self.map_height -1 - i + 0.5) * self.resolution
    
        return x_world, y_world

    def a_star_path_planner(self, start_pose, end_pose):
        """
        Plans a path using A* from start_pose to end_pose.
        Args:
            - start_pose : PoseStamped start position
            - end_pose : PoseStamped goal position
        Returns:
            - path : Path object containing waypoints of the planned path
        """
        path = Path()
        
        self.get_logger().info(
            'A* planner.\n> start: {},\n> end: {}'.format(start_pose.pose.position, end_pose.pose.position))
        self.start_time = self.get_clock().now().nanoseconds*1e-9 #Do not edit this line (required for autograder)
        
        # TODO: IMPLEMENTATION OF THE A* ALGORITHM
        start_i,start_j = self.pose_to_grid(start_pose)
        end_i,end_j = self.pose_to_grid(end_pose)
        start_node_name = '%d,%d' %(start_i,start_j)
        end_node_name = '%d,%d' %(end_i,end_j)
        
        start_node = self.map_processor.map_graph.g[start_node_name]
        end_node = self.map_processor.map_graph.g[end_node_name]
        self.map_processor.map_graph.root = start_node_name
        self.map_processor.map_graph.end = end_node_name
        self.get_logger().info('Replanning triggered!')

        # Run A* on the graph
        astar = AStar(self.map_processor.map_graph)
        astar.solve(start_node,end_node)
        path_grid,dist = astar.reconstruct_path(start_node,end_node)
        self.get_logger().info('A* found path with %d waypoints, distance: %.2f' % (len(path_grid), dist))
        
        # Print path coordinates and display path
        self.map_processor.path_coords(path_grid)
        self.map_processor.draw_path_graph(path_grid)

        # Build ROS Path message
        path.header.frame_id = "map"
        path.header.stamp = self.get_clock().now().to_msg()
    
        for node_name in path_grid:
            i, j = map(int, node_name.split(','))
            x_world,y_world = self.grid_to_pose(i, j)
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.pose.position.x = x_world
            pose.pose.position.y = y_world
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 0.0
            path.poses.append(pose)
        
        self.path_pub.publish(path)

        # Do not edit below (required for autograder)
        self.astarTime = Float32()
        self.astarTime.data = float(self.get_clock().now().nanoseconds*1e-9-self.start_time)
        self.calc_time_pub.publish(self.astarTime)
        
        return path

    def get_path_idx(self, path, vehicle_pose):
        """
        Computes the next waypoint index along the path for the robot to follow.
        Args:
            - path : Path object containing waypoints
            - vehicle_pose : current PoseStamped of the robot
        Returns:
            - lookahead_idx : index of next waypoint to follow
        """
        min_lookahead = 0.1   # meters (close to goal)
        max_lookahead = 0.4   # meters (far from goal)
        goal_threshold = 0.75
        closest_idx = 0
        min_dist = float('inf')
        x_robot = vehicle_pose.pose.position.x
        y_robot = vehicle_pose.pose.position.y

        qx = vehicle_pose.pose.orientation.x
        qy = vehicle_pose.pose.orientation.y
        qz = vehicle_pose.pose.orientation.z
        qw = vehicle_pose.pose.orientation.w
        yaw_robot = math.atan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))
        
        closest_idx = 0
        min_dist = float('inf') 

        x_goal_final = path.poses[-1].pose.position.x
        y_goal_final = path.poses[-1].pose.position.y
        dist_to_final_goal = math.hypot(x_goal_final - x_robot, y_goal_final - y_robot)

        # Adaptive lookahead
        if dist_to_final_goal > goal_threshold:
            lookahead_distance = max_lookahead
        else:
            lookahead_distance = min_lookahead + (max_lookahead - min_lookahead) * (dist_to_final_goal / goal_threshold)
        
        # Find closest waypoint in front of the robot
        for i, pose in enumerate(path.poses):
            x_wp = pose.pose.position.x
            y_wp = pose.pose.position.y
            
            # Vector from robot to waypoint
            dx = x_wp - x_robot
            dy = y_wp - y_robot
            dist = math.sqrt(dx**2 + dy**2)
            robot_forward_x = math.cos(yaw_robot)
            robot_forward_y = math.sin(yaw_robot)
            
            # Dot product to check if waypoint is in front
            dot_product = dx * robot_forward_x + dy * robot_forward_y
            
            # Only consider waypoints ahead of the robot
            if dot_product > 0 and dist < min_dist:
                min_dist = dist
                closest_idx = i

        # Find index of lookahead point
        lookahead_idx = closest_idx
        for i in range(closest_idx, len(path.poses)):
            x_wp = path.poses[i].pose.position.x
            y_wp = path.poses[i].pose.position.y
            dist = math.sqrt((x_robot - x_wp)**2 + (y_robot - y_wp)**2)
            if dist >= lookahead_distance:
                lookahead_idx = i
                break

        # Save and return the chosen index
        self.last_path_idx = lookahead_idx
        x_goal_final = path.poses[-1].pose.position.x
        y_goal_final = path.poses[-1].pose.position.y
        self.dist_to_final_goal = math.sqrt((x_robot - x_goal_final)**2 + (y_robot - y_goal_final)**2)

        self.get_logger().info(
        f"[FOLLOWER] Closest idx: {closest_idx}, Lookahead idx: {lookahead_idx}, "
        f"Dist to next wp: {min_dist:.3f}, Dist to final goal: {self.dist_to_final_goal:.3f}")
        return lookahead_idx

    def path_follower(self, vehicle_pose, current_goal_pose):
        """
        Computes distance and heading error to current goal waypoint.
        Args:
            - vehicle_pose : current PoseStamped of robot
            - current_goal_pose : PoseStamped of next goal along path
        Returns:
            - dist : distance to waypoint
            - heading_error : heading error in radians
            - yaw_robot : current robot yaw in radians
        """
 
        # TODO: IMPLEMENT PATH FOLLOWER
        x_robot = vehicle_pose.pose.position.x
        y_robot = vehicle_pose.pose.position.y
        x_goal = current_goal_pose.pose.position.x
        y_goal = current_goal_pose.pose.position.y

        # Euclidean distance
        dist = math.sqrt((x_robot - x_goal) ** 2 + (y_robot - y_goal) ** 2)
        desired_heading = math.atan2((y_goal - y_robot),(x_goal - x_robot))

        qx = vehicle_pose.pose.orientation.x
        qy = vehicle_pose.pose.orientation.y
        qz = vehicle_pose.pose.orientation.z
        qw = vehicle_pose.pose.orientation.w
        yaw_robot = math.atan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))
        
        heading_error = desired_heading - yaw_robot
        heading_error = math.atan2(math.sin(heading_error), math.cos(heading_error))

        self.get_logger().info(f"x_goal={x_goal:.4f}, y_goal={y_goal:.4f}, x_robot={x_robot:.4f}, y_robot={y_robot:.4f}, desired_heading={math.degrees(desired_heading):.4f}, \
                               yaw_robot={math.degrees(yaw_robot):.4f}, heading_error={math.degrees(heading_error):.4f}")

        return dist, heading_error, yaw_robot

    def move_ttbot(self, distance, heading_error):
        """
        Computes linear and angular velocity commands and publishes them to move the robot.
        Args:
            - distance : distance to next waypoint
            - heading_error : heading error to waypoint
        Returns:
            - None
        """
        cmd_vel = Twist()
        # TODO: IMPLEMENT YOUR LOW-LEVEL CONTROLLER
        kp_lin = 0.25
        ki_lin = 0.05
        kd_lin = 0.35

        kp_ang = 0.25
        ki_ang = 0.05
        kd_ang = 0.75

        max_speed = 0.15
        max_ang_speed = 1.0
        dt = 0.1

        if not hasattr(self, "prev_lin_error"):
            self.prev_lin_error = 0.0
            self.integral_lin = 0.0
            self.prev_ang_error = 0.0
            self.integral_ang = 0.0

        # Compute commands
        lin_error = distance
        self.integral_lin += lin_error * dt
        derivative_lin = (lin_error - self.prev_lin_error) / dt

        # Linear PID
        linear_speed = (
            kp_lin * lin_error +
            ki_lin * self.integral_lin * 0 +
            kd_lin * derivative_lin
        )
        linear_speed = max(min(linear_speed, max_speed), -max_speed)

        ang_error = heading_error
        self.integral_ang += ang_error * dt
        derivative_ang = (ang_error - self.prev_ang_error) / dt

        # Angular PID
        angular_speed = (
            kp_ang * ang_error +
            ki_ang * self.integral_ang * 0+
            kd_ang * derivative_ang 
        )
        angular_speed = max(min(angular_speed, max_ang_speed), -max_ang_speed)
        
        self.prev_lin_error = lin_error
        self.prev_ang_error = ang_error

        # If heading error is large, rotate in place
        heading_threshold = math.radians(30)  # ~30 degrees
        if abs(heading_error) > heading_threshold:
            self.get_logger().info(f"[TURN-IN-PLACE] Large heading error ({math.degrees(heading_error):.2f}°) → rotating only")
            linear_speed = 0.0  # stop forward motion

        cmd_vel = Twist()
        cmd_vel.linear.x = linear_speed
        cmd_vel.angular.z = angular_speed

        self.get_logger().info(f"[MOVE] Linear Speed: {linear_speed:.3f}, Angular Speed: {angular_speed:.3f}")
        self.cmd_vel_pub.publish(cmd_vel)

    def run(self):
        """
        Main control loop. Plans path, follows it, and commands the robot until goal is reached.
        Args:
            - None
        Returns:
            - None
        """
        path = None
        self.replan_needed = True
        self.path_idx = 0
        goal_tolerance = 0.1  # meters
        
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)

            # Skip loop if robot or goal is uninitialized
            if (self.ttbot_pose.pose.position.x == 0.0 and self.ttbot_pose.pose.position.y == 0.0) or \
            (self.goal_pose.pose.position.x == 0.0 and self.goal_pose.pose.position.y == 0.0):
                continue  

            # Plan only once
            if self.replan_needed:
                path = self.a_star_path_planner(self.ttbot_pose, self.goal_pose)
                self.replan_needed = False
                self.path_idx = 0

            # Get next goal waypoint    
            self.path_idx = self.get_path_idx(path, self.ttbot_pose)
            current_goal = path.poses[self.path_idx]
            dist, heading_error,yaw_robot = self.path_follower(self.ttbot_pose, current_goal)

            # Stop if goal reached
            if self.dist_to_final_goal < goal_tolerance:
                self.get_logger().info("Final goal reached. Holding position.")
                self.goal_reached = True
                cmd_vel = Twist()
                cmd_vel.linear.x = 0.0
                cmd_vel.angular.z = 0.0
                self.cmd_vel_pub.publish(cmd_vel)
                continue

            if self.goal_reached:
                continue
            
            # Move robot 
            self.move_ttbot(dist, heading_error)
            self.get_logger().info("[RUN] Tick loop alive") #del
        
class Queue():
    def __init__(self, init_queue = []):
        '''
        Initializes a queue data structure used for A* search bookkeeping.
        Args:
            - init_queue (list): Initial list of nodes to populate the queue.
        Returns:
            - None
        '''
        self.queue = copy(init_queue)
        self.start = 0
        self.end = len(self.queue)-1

    def __len__(self):
        numel = len(self.queue)
        return numel

    def __repr__(self):
        '''
        Provides a readable string representation of the queue for debugging.
        Args:
            - None
        Returns:
            - tmpstr (str): Formatted string showing queue contents and boundaries.
        '''
        q = self.queue
        tmpstr = ""
        for i in range(len(self.queue)):
            flag = False
            if i == self.start:
                tmpstr += "<"
                flag = True
            if i == self.end:
                tmpstr += ">"
                flag = True

            if flag:
                tmpstr += '| ' + str(q[i]) + '|\n'
            else:
                tmpstr += ' | ' + str(q[i]) + '|\n'

        return tmpstr

    def __call__(self):
        return self.queue

    def initialize_queue(self, init_queue = []):
        '''
        Resets the queue with a new list of nodes.
        '''
        self.queue = copy(init_queue)
        self.start = 0
        self.end = len(self.queue) - 1

    def sort(self, key=str.lower):
        '''
        Sorts the queue in-place based on a provided key function.
        '''
        self.queue = sorted(self.queue, key=key)

    def push(self, data):
        '''
        Appends a new node to the end of the queue.
        '''
        self.queue.append(data)
        self.end += 1

    def pop(self):
        '''
        Removes and returns the node at the front of the queue.
        '''
        p = self.queue.pop(self.start)
        self.end = len(self.queue) - 1
        return p
    
class AStar():
    def __init__(self,in_tree):
        """
        Args:
            - in_tree : Graph/tree structure 
        Returns:
            - None
        """
        self.in_tree = in_tree
        self.q = Queue()
        self.dist = {name:np.inf for name,node in in_tree.g.items()}  # g score
        self.h = {name:0 for name,node in in_tree.g.items()}  # h score 

        #Compute Euclidean distance for each node
        for name,node in in_tree.g.items():
            start = tuple(map(int, name.split(',')))
            end = tuple(map(int, self.in_tree.end.split(',')))
            self.h[name] = np.sqrt((end[0]-start[0])**2 + (end[1]-start[1])**2)

        # Dictionary storing predecessor of each node in the shortest path
        self.via = {name:0 for name,node in in_tree.g.items()}

        #Push all nodes into queue
        for __,node in in_tree.g.items():
            self.q.push(node)

    def __get_f_score(self,node):
        """
        Computes the f score of a node
        Args:
            - node : node whose f-score is being computed
        Returns:
            - f_score : f = g + h for the given node
        """
        return self.dist[node.name] + self.h[node.name]

    def solve(self, sn, en):
        """
        Runs A* from start node to end node
        Args:
            - sn : start node
            - en : goal node
        Returns:
            - None
        """
        self.dist[sn.name] = 0
        open_set = [sn]
        closed_set = set()

        while open_set:
            # Select node in open set with the minimum f-score
            u = min(open_set, key=self.__get_f_score)
            open_set.remove(u)

             # If goal is reached, terminate search
            if u.name == en.name:
                break
            
            # Mark current node as explored
            closed_set.add(u.name)

            for i in range(len(u.children)):
                c = u.children[i]
                w = u.weight[i]
                if c.name in closed_set:
                    continue

                new_dist = self.dist[u.name] + w
                if new_dist < self.dist[c.name]:
                    self.dist[c.name] = new_dist
                    self.via[c.name] = u.name
                    if c not in open_set:
                        open_set.append(c)

    def reconstruct_path(self,sn,en):
        """
        Reconstructs the shortest path from the start node to the goal node
        Args:
            - sn : start node
            - en : goal node
        Returns:
            - path : ordered list of node names from start to goal
            - dist :total cost of the path
        """
        end_key = en.name
        u = end_key
        path = [u]
        dist = self.dist[end_key]
        start_key = sn.name

        # Backtrack using the via dictionary until the start node is reached
        while u!=start_key:
          u = self.via[u]
          path.append(u)

        path.reverse()

        return path,dist

class Node():
    def __init__(self,name):
        """
        Args:
            - name : unique identifier for the node
        Returns:
            - None
        """
        self.name = name
        self.children = []
        self.weight = []  # List of edge weights

    def __repr__(self):
        """
        Args:
            - None
        Returns:
            - name : string representation of the node
        """
        return self.name

    def add_children(self,node,w=None):
        """
        Adds child nodes and associated edge weights to the current node.
        Args:
            - node : list of child nodes to connect
            - w : edge weights corresponding to each child.
                If None, all edge weights default to 1
        Returns:
            - None
        """
        if w == None:
            w = [1]*len(node)
        
        # Update child nodes and edge weight lists 
        self.children.extend(node)
        self.weight.extend(w)

class Tree():
    def __init__(self,name):
        """
        Args:
            - name : identifier for the tree/graph
        Returns:
            - None
        """
        self.name = name
        self.root = 0
        self.end = 0
        self.g = {}

    def add_node(self, node, start = False, end = False):
        """
        Adds a node to the graph and optionally marks it as the start or end node.
        Args:
            - node : node to be added to the tree
            - start : if True, mark this node as the root
            - end : if True, mark this node as the end node
        Returns:
            - None
        """
        self.g[node.name] = node
        if(start):
            self.root = node.name
        elif(end):
            self.end = node.name

    def set_as_root(self,node):
        """
        Marks a node as the root of the graph.
        Args:
            - node : node to be set as the root
        Returns:
            - None
        """
        self.root = True
        self.end = False

    def set_as_end(self,node):
        """
        Marks a node as the end of the graph.
        Args:
            - node : node to be set as the end
        Returns:
            - None
        """
        self.root = False
        self.end = True

class Map():
    def __init__(self, map_name):
        """
        Args:
            - map_name : name of the map file
        Returns:
            - None
        """
        self.map_im, self.map_df, self.limits = self.__open_map(map_name)
        self.image_array = self.__get_obstacle_map(self.map_im, self.map_df)

    def __repr__(self):
        """
        Displays the obstacle map using matplotlib
        Args:
            - None
        Returns:
            - str: empty string 
        """
        fig, ax = plt.subplots(dpi=150)
        ax.imshow(self.image_array,extent=self.limits, cmap=cm.gray)
        ax.plot()
        return ""

    def __open_map(self,map_name):
        """
        Reads the map YAML file and image, converts the image to grayscale, and computes world-coordinate limits.
        Args:
            - map_name : base name of the map YAML file
        Returns:
            - im : grayscale map image
            - map_df : map metadata loaded from YAML
            - limits : world-coordinate bounds
        """
        f = open(map_name + '.yaml', 'r')
        map_df = pd.json_normalize(yaml.safe_load(f))
        map_name = map_df.image[0]
        im = Image.open(map_name)
        #size = 200, 200
        #im.thumbnail(size)
        im = ImageOps.grayscale(im)

        # Compute map bounds
        xmin = map_df.origin[0][0]
        xmax = map_df.origin[0][0] + im.size[0] * 0+ map_df.resolution[0]
        ymin = map_df.origin[0][1]
        ymax = map_df.origin[0][1] + im.size[1] * 0+  map_df.resolution[0]

        return im, map_df, [xmin,xmax,ymin,ymax]

    def __get_obstacle_map(self,map_im, map_df):
        """
        Processes the grayscale map image into a binary obstacle map based on occupancy thresholds.
        Args:
            - map_im : grayscale map image
            - map_df  map metadata containing thresholds
        Returns:
            - img_array : binary obstacle map (0 = free, 255 = occupied)
        """
        img_array = np.reshape(list(self.map_im.getdata()),(self.map_im.size[1],self.map_im.size[0]))
        
        # Convert occupancy thresholds from [0,1] to [0,255]
        up_thresh = self.map_df.occupied_thresh[0]*255
        low_thresh = self.map_df.free_thresh[0]*255

        for j in range(self.map_im.size[0]):
            for i in range(self.map_im.size[1]):
                if img_array[i,j] > up_thresh:
                    img_array[i,j] = 255
                else:
                    img_array[i,j] = 0
        
        return img_array

class MapProcessor():
    def __init__(self,name):
        """
        Args:
            - name : base name of the map files
        Returns:
            - None
        """
        self.map = Map(name)
        self.inf_map_img_array = np.zeros(self.map.image_array.shape)
        self.map_graph = Tree(name)

    def __modify_map_pixel(self,map_array,i,j,value,absolute):
        """
        Modifies a pixel in a map array.
        Args:
            - map_array : numpy array representing the map
            - i : row index
            - j : column index
            - value : value to assign or add
            - absolute : if True, overwrite; else add value
        Returns:
            - None
        """
        if( (i >= 0) and
            (i < map_array.shape[0]) and
            (j >= 0) and
            (j < map_array.shape[1]) ):
            if absolute:
                map_array[i][j] = value
            else:
                map_array[i][j] += value

    def __inflate_obstacle(self,kernel,map_array,i,j,absolute):
        """
        Inflates an obstacle at (i,j) using the specified kernel.
        Args:
            - kernel : 2D numpy array used for inflation
            - map_array : map being modified
            - i : row index of obstacle
            - j : column index of obstacle
            - absolute : if True, overwrite; else add values
        Returns:
            - None
        """
        dx = int(kernel.shape[0]//2)
        dy = int(kernel.shape[1]//2)
        if (dx == 0) and (dy == 0):
            self.__modify_map_pixel(map_array,i,j,kernel[0][0],absolute)
        else:
            for k in range(i-dx,i+dx):
                for l in range(j-dy,j+dy):
                    self.__modify_map_pixel(map_array,k,l,kernel[k-i+dx][l-j+dy],absolute)

    def inflate_map(self,kernel,absolute=True):
        """
        Inflates obstacles in the map using the given kernel.
        Args:
            - kernel : 2D numpy array used to inflate obstacles
            - absolute : if True, overwrite values; else add
        Returns:
            - None
        """
        self.inf_map_img_array = np.zeros(self.map.image_array.shape)
        for i in range(self.map.image_array.shape[0]):
            for j in range(self.map.image_array.shape[1]):
                if self.map.image_array[i][j] == 0:
                    self.__inflate_obstacle(kernel,self.inf_map_img_array,i,j,absolute)

        # normalize inflated map to [0,1]
        r = np.max(self.inf_map_img_array)-np.min(self.inf_map_img_array)
        if r == 0:
            r = 1
        self.inf_map_img_array = (self.inf_map_img_array - np.min(self.inf_map_img_array))/r

    def get_graph_from_map(self):
        """
        Converts free cells in the inflated map into graph nodes and connects neighbors.
        Args:
            - None
        Returns:
            - None
        """
        for i in range(self.map.image_array.shape[0]):
            for j in range(self.map.image_array.shape[1]):
                if self.inf_map_img_array[i][j] == 0:
                    node = Node('%d,%d'%(i,j))
                    self.map_graph.add_node(node)
        # Connect the nodes through edges
        for i in range(self.map.image_array.shape[0]):
            for j in range(self.map.image_array.shape[1]):
                if self.inf_map_img_array[i][j] == 0:
                    if (i > 0):
                        if self.inf_map_img_array[i-1][j] == 0:
                            # add an edge up
                            child_up = self.map_graph.g['%d,%d'%(i-1,j)]
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_up],[1])
                    if (i < (self.map.image_array.shape[0] - 1)):
                        if self.inf_map_img_array[i+1][j] == 0:
                            # add an edge down
                            child_dw = self.map_graph.g['%d,%d'%(i+1,j)]
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_dw],[1])
                    if (j > 0):
                        if self.inf_map_img_array[i][j-1] == 0:
                            # add an edge to the left
                            child_lf = self.map_graph.g['%d,%d'%(i,j-1)]
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_lf],[1])
                    if (j < (self.map.image_array.shape[1] - 1)):
                        if self.inf_map_img_array[i][j+1] == 0:
                            # add an edge to the right
                            child_rg = self.map_graph.g['%d,%d'%(i,j+1)]
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_rg],[1])
                    if ((i > 0) and (j > 0)):
                        if self.inf_map_img_array[i-1][j-1] == 0:
                            # add an edge up-left
                            child_up_lf = self.map_graph.g['%d,%d'%(i-1,j-1)]
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_up_lf],[np.sqrt(2)])
                    if ((i > 0) and (j < (self.map.image_array.shape[1] - 1))):
                        if self.inf_map_img_array[i-1][j+1] == 0:
                            # add an edge up-right
                            child_up_rg = self.map_graph.g['%d,%d'%(i-1,j+1)]
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_up_rg],[np.sqrt(2)])
                    if ((i < (self.map.image_array.shape[0] - 1)) and (j > 0)):
                        if self.inf_map_img_array[i+1][j-1] == 0:
                            # add an edge down-left
                            child_dw_lf = self.map_graph.g['%d,%d'%(i+1,j-1)]
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_dw_lf],[np.sqrt(2)])
                    if ((i < (self.map.image_array.shape[0] - 1)) and (j < (self.map.image_array.shape[1] - 1))):
                        if self.inf_map_img_array[i+1][j+1] == 0:
                            # add an edge down-right
                            child_dw_rg = self.map_graph.g['%d,%d'%(i+1,j+1)]
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_dw_rg],[np.sqrt(2)])

    def gaussian_kernel(self, size, sigma=1):
        """
        Generates a normalized Gaussian kernel for obstacle inflation.
        Args:
            - size : kernel size 
            - sigma : standard deviation of the Gaussian
        Returns:
            - sm : normalized 2D numpy array representing the kernel
        """
        size = int(size) // 2
        x, y = np.mgrid[-size:size+1, -size:size+1]
        normal = 1 / (2.0 * np.pi * sigma**2)
        g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal

        # normalize to [0,1]
        r = np.max(g)-np.min(g)
        sm = (g - np.min(g))*1/r
        return sm

    def rect_kernel(self, size, value):
        """
        Generates a rectangular kernel for obstacle inflation.
        Args:
            - size : kernel dimension 
            - value : unused
        Returns:
            - m : 2D numpy array of ones
        """
        m = np.ones(shape=(size,size))
        return m

    def path_coords(self, path):
        """
        Converts a path list to coordinates and overlays it on the inflated map.
        Args:
            - path : list of node names along the path
        Returns:
            - path_array : map with path 
        """
        path_tuple_list = []
        path_array = copy(self.inf_map_img_array)

        print("\n=== PATH COORDINATES ===")
        for idx in path:
            tup = tuple(map(int, idx.split(',')))
            path_tuple_list.append(tup)
            path_array[tup] = 0.5
            print(tup)  # print each coordinate

        print("=== END OF PATH ===\n")
        return path_array

    def draw_path_graph(self, path):
        """
        Visualizes the A* path on the inflated map using matplotlib.
        Args:
            - path : list of node names along the path
        Returns:
            - None
        """
        map_img = copy(self.inf_map_img_array)
        res = self.map.map_df.resolution[0]
        origin_x, origin_y = self.map.map_df.origin[0][:2]  # only take x, y
        height = map_img.shape[0]

        path_coords = [tuple(map(int, node.split(','))) for node in path]
        path_i = [i for (i,j) in path_coords]
        path_j = [j for (i,j) in path_coords]

        plt.figure(figsize=(8,8))
        plt.imshow(map_img, cmap='gray', origin='upper')
        plt.plot(path_j, path_i, color='red', linewidth=2)
        plt.scatter(path_j[0], path_i[0], color='green', s=50, label='Start')
        plt.scatter(path_j[-1], path_i[-1], color='blue', s=50, label='Goal')
        plt.title("A* Planned Path on Map")
        plt.xlabel("Columns (X)")
        plt.ylabel("Rows (Y)")
        plt.legend()
        plt.show()

def main(args=None):
    rclpy.init(args=args)
    mp = MapProcessor("/home/me597/lab3_ws/3/task_4/maps/classroom_map")
    kernel = mp.rect_kernel(5, 1)   # or mp.gaussian_kernel(3, sigma=1)
    mp.inflate_map(kernel, absolute=True)   # fills mp.inf_map_img_array
    mp.get_graph_from_map() 
    nav = Navigation(mp,node_name='Navigation')

    try:
        nav.run()
    except KeyboardInterrupt:
        pass
    finally:
        nav.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()