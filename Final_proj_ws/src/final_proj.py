"""
Author: Aryaman Akolkar
Course: ME 597, Purdue University
Semester: Fall 2025

Final Project
Goal: Implement a SLAM algorithm that enables the robot to navigate an unknown environment
NOTE:
All classes in this file, except for the Task1 class, were provided by the course instructors. 
The MapProcessor, Queue, Node, Tree, and AStar classes are instructor-supplied; 
however, the solve function within the AStar class, along with the entire Task1 class, 
was fully implemented by the author
"""

#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import yaml
from copy import copy, deepcopy
import math
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path, OccupancyGrid, Odometry
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Pose, Twist
from std_msgs.msg import Float32
from visualization_msgs.msg import Marker, MarkerArray

class Task(Node):
    def __init__(self):
        super().__init__('task1_node')
        self.timer = self.create_timer(0.5, self.main_loop)
        self.map_ready = False
        self.map_updated = False
        self.path = Path()
        self.goal_pose = PoseStamped()
        self.ttbot_pose = PoseStamped()
        self.current_odom = Odometry()
        self.start_time = 0.0
        self.last_path_idx = 0
        self.goal_reached = False
        self.current_goal = None
        self.goal_locked = False
                
        # Subscribers
        self.create_subscription(OccupancyGrid, "/map", self.map_create_cbk, 10)
        self.create_subscription(Odometry, '/odom', self.odom_cbk, 10)

        # Publishers
        self.marker_pub = self.create_publisher(MarkerArray, "/frontier_markers", 10)
        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.path_pub = self.create_publisher(Path, 'global_plan', 10)
        self.calc_time_pub = self.create_publisher(Float32, 'astar_time',10) #DO NOT MODIFY

    def main_loop(self):
        """
        Callback function to ensure code is running
        """
        self.get_logger().info('Task1 node is alive.', throttle_duration_sec=1)

    def map_create_cbk(self,msg):
        """
        Callback function for receiving and processing the occupancy grid map.
        Args:
            - msg : nav_msgs.msg.OccupancyGrid message containing the map data
        Returns:
            - None
        """
        new_map = np.array(msg.data)

        if hasattr(self, "last_map") and np.array_equal(new_map, self.last_map):
            return

        self.last_map = new_map.copy()
        self.replan_needed = True
        self.map_width  = msg.info.width
        self.map_height = msg.info.height
        self.resolution = msg.info.resolution
        self.map_data   = new_map.reshape((self.map_height, self.map_width))
        self.map_data = self.inflate_obstacles(radius_cells=4)
        self.x_min = msg.info.origin.position.x
        self.y_min = msg.info.origin.position.y
        self.map_ready = True
        self.remove_surrounded_unknowns()

        self.frontier_cells = self.frontier()
        self.frontier_markers(self.frontier_cells)

    def odom_cbk(self,msg):
        """
            Callback function for calculating robot pose.
        Args:
            - msg : nav_msgs.msg.Odometry message containing robot pose information
        Returns:
            - None
        """
        robot_pose = msg
        self.x_robot = robot_pose.pose.pose.position.x
        self.y_robot = robot_pose.pose.pose.position.y
        q = robot_pose.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y*q.y + q.z*q.z)
        self.yaw = math.atan2(siny_cosp, cosy_cosp)
        if robot_pose is not None:
            self.get_logger().info(f"Robot Pose: {self.x_robot:.2f}, {self.y_robot:.2f}, yaw_z={math.degrees(self.yaw):.2f}",throttle_duration_sec=1)
        else:
            self.get_logger().info("Robot pose not available yet.",throttle_duration_sec=1)

    def remove_surrounded_unknowns(self):
        """
            Cleans the occupancy grid by converting surrounding unknown cells (-1) into free cells
        Args:
            - None
        Returns:
            - None
        """
        cleaned = self.map_data.copy()
        neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1),(1, 1), (1, -1), (-1, 1), (-1, -1)]

        for i in range(self.map_height):
            for j in range(self.map_width):

                # Only process unknown cells
                if self.map_data[i, j] != -1:
                    continue

                surrounded = True
                for dy, dx in neighbors:
                    ni = i + dy
                    nj = j + dx
                    # out of bounds:treat as not surrounded
                    if not (0 <= ni < self.map_height and 0 <= nj < self.map_width):
                        surrounded = False
                        break
                    val = self.map_data[ni, nj]
                    # unknown or inflated: not known
                    if val == -1 or (1 <= val < 100):
                        surrounded = False
                        break
                if surrounded:
                    cleaned[i, j] = 0
        self.map_data = cleaned
        
    def frontier(self):
        """
            Identifies frontier cells in the occupancy grid.
        Args:
            - None
        Returns:
            - frontier_cells : list of (row, column) tuples representing frontier cells
        """
        if not self.map_ready:
            return []
        frontier_cells = []
        neighbors = [(1,0),(0,1),(-1,0),(0,-1)] 
        for i in range(self.map_height):
            for j in range(self.map_width):

                # Only consider free cells
                if self.map_data[i,j] != 0.0:
                    continue

                has_unknown = False
                has_obstacle = False

                # Check neighboring cells
                for a,b in neighbors:
                    ni = i + a     
                    nj = j + b
                    
                    if 0 <= ni < self.map_height and 0 <= nj < self.map_width:
                        if self.map_data[ni,nj] == -1:   #Checks if neighboring cell is unknown
                            has_unknown = True
                        elif self.map_data[ni,nj] == 100:
                            has_obstacle = True

                # Frontier cell criteria
                if has_unknown and not has_obstacle:
                    frontier_cells.append((i,j))

        return frontier_cells
    
    def closest_cell(self,frontier_cells):
        """
        Selects the closest valid frontier cell to the robot in world coordinates.
        Args:
            - frontier_cells : list of tuples representing frontier cells
        Returns:
            - best : tuple of the closest frontier cell in world coordinates
        """
        min_dist = float('inf')
        for (r, c) in frontier_cells:
            x_world = (c + 0.5) * self.resolution + self.x_min
            y_world = (r + 0.5) * self.resolution + self.y_min

            d = np.hypot(x_world - self.x_robot, y_world - self.y_robot)
            
            # Skip frontier cells that are too close to the robot
            if d < 1.0:
                continue
            
            # Update closest frontier cell
            if d < min_dist:
                min_dist = d
                best = (x_world, y_world)

        return best        
    
    def frontier_markers(self,frontier_cells):
        """
        Publishes visualization markers for all detected frontier cells.
        Args:
            - frontier_cells : list of tuples representing frontier cells
        Returns:
            - None
        """
        marker_array = MarkerArray()
        m_id = 0

        #Frontier cells (green points) 
        for (r, c) in frontier_cells:
            frontier_marker = Marker()
            frontier_marker.header.frame_id = "map"
            frontier_marker.ns = "frontier_cells"
            frontier_marker.type = Marker.CUBE
            frontier_marker.action = Marker.ADD
            
            #  Marker size
            frontier_marker.scale.x = 0.05
            frontier_marker.scale.y = 0.05
            frontier_marker.scale.z = 0.01
            
            # Marker colour
            frontier_marker.color.r = 0.0
            frontier_marker.color.g = 1.0
            frontier_marker.color.b = 0.0
            frontier_marker.color.a = 1.0
            frontier_marker.id = m_id
            m_id += 1

            # Convert to world coordinates
            frontier_marker.pose.position.x = (c + 0.5) * self.resolution + self.x_min
            frontier_marker.pose.position.y = (r + 0.5) * self.resolution + self.y_min
            frontier_marker.pose.position.z = 0.0
            
            marker_array.markers.append(frontier_marker)

        self.marker_pub.publish(marker_array)
    
    def a_star_path_planner(self, x_robot, y_robot, x_goal,y_goal):
        """
        Plans a path using A* from start_pose to end_pose.
        Args:
            - start_pose : PoseStamped start position
            - end_pose : PoseStamped goal position
        Returns:
            - path : Path object containing waypoints of the planned path
        """
        path = Path()
        
        # TODO: IMPLEMENTATION OF THE A* ALGORITHM
        x_robot_map = int((x_robot - self.x_min) / self.resolution)
        y_robot_map = int((y_robot - self.y_min) / self.resolution)

        x_goal_map = int((x_goal - self.x_min) / self.resolution)
        y_goal_map = int((y_goal - self.y_min) / self.resolution)
        
        start_name = f"{y_robot_map},{x_robot_map}"
        end_name   = f"{y_goal_map},{x_goal_map}"

        # If goal not in graph, pick the nearest frontier cell that is in graph
        while start_name not in self.map_processor.map_graph.g:
            self.get_logger().warn("Waiting for valid start…")
            rclpy.spin_once(self, timeout_sec=1)

            x_robot_map = int((x_robot - self.x_min) / self.resolution)
            y_robot_map = int((y_robot - self.y_min) / self.resolution)
            start_name = f"{y_robot_map},{x_robot_map}"
        start_name = f"{y_robot_map},{x_robot_map}"
        end_name   = f"{y_goal_map},{x_goal_map}"

        start_node = self.map_processor.map_graph.g[start_name]
        end_node   = self.map_processor.map_graph.g[end_name]

        astar = AStar(self.map_processor.map_graph)
        astar.solve(start_node, end_node)
        path_grid,dist = astar.reconstruct_path(start_node,end_node)
        self.get_logger().info('A* found path with %d waypoints, distance: %.2f' % (len(path_grid), dist))

        path.header.frame_id = "map"
        path.header.stamp = self.get_clock().now().to_msg()

        for r, c in path_grid:
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            # Convert grid back to world coordinates
            pose.pose.position.x = (c + 0.5) * self.resolution + self.x_min
            pose.pose.position.y = (r + 0.5) * self.resolution + self.y_min
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0  
            path.poses.append(pose)

        path.header.frame_id = "map"
    
        # Publish the path for RViz visualization
        self.path_pub.publish(path)
        self.get_logger().info("Path planned.")
        
        return path

    def get_path_idx(self, path, ):
        """
        Computes the next waypoint index along the path for the robot to follow.
        Args:
            - path : Path object containing waypoints
        Returns:
            - lookahead_idx : index of next waypoint to follow
        """
        min_lookahead = 0.1  # meters (close to goal)
        max_lookahead = 0.4   # meters (far from goal)
        goal_threshold = 0.75 #Changed from 0.75
        closest_idx = 0
        min_dist = float('inf')
        x_robot = self.x_robot
        y_robot = self.y_robot

        yaw_robot = self.yaw
        
        closest_idx = 0
        min_dist = float('inf') 

        x_goal_final = path.poses[-1].pose.position.x
        y_goal_final = path.poses[-1].pose.position.y
        dist_to_final_goal = math.hypot(x_goal_final - x_robot, y_goal_final - y_robot)

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

        return lookahead_idx

    def path_follower(self, current_goal_pose):
        """
        Computes distance and heading error to current goal waypoint.
        Args:
            - current_goal_pose : PoseStamped of next goal along path
        Returns:
            - dist : distance to waypoint
            - heading_error : heading error in radians
            - yaw_robot : current robot yaw in radians
        """
  
        # TODO: IMPLEMENT PATH FOLLOWER
        x_robot = self.x_robot
        y_robot = self.y_robot
        x_goal = current_goal_pose.pose.position.x
        y_goal = current_goal_pose.pose.position.y
        self.get_logger().info(f'x_goal:{x_goal:.2f},y_goal:{y_goal:.2f}')

        dist = math.sqrt((x_robot - x_goal) ** 2 + (y_robot - y_goal) ** 2)
        desired_heading = math.atan2((y_goal - y_robot),(x_goal - x_robot))
        yaw_robot = self.yaw
        
        heading_error = desired_heading - yaw_robot
        self.get_logger().info(f'Desired Heading:{math.degrees(desired_heading):.2f},Yaw Robot:{math.degrees(yaw_robot):.2f}')
        heading_error = math.atan2(math.sin(heading_error), math.cos(heading_error))

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
        kp_lin = 1.0
        ki_lin = 0.05
        kd_lin = 1.0

        kp_ang = 0.75
        ki_ang = 0.05
        kd_ang = 0.5

        max_speed = 0.45
        max_ang_speed = 0.5
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
            ki_ang * self.integral_ang * 0 +
            kd_ang * derivative_ang 
        )
        angular_speed = max(min(angular_speed, max_ang_speed), -max_ang_speed)
    
        self.prev_lin_error = lin_error
        self.prev_ang_error = ang_error

        # If heading error is large, rotate in place
        heading_threshold = math.radians(20)  # ~20 degrees
        if abs(heading_error) > heading_threshold:
            self.get_logger().info(f"[TURN-IN-PLACE] Large heading error ({math.degrees(heading_error):.2f}°) → rotating only")
            linear_speed = 0.0  # stop forward motion
       
        cmd_vel = Twist()
        cmd_vel.linear.x = linear_speed
        cmd_vel.angular.z = angular_speed

        self.get_logger().info(f"[MOVE] Linear Speed: {linear_speed:.3f}, Angular Speed: {angular_speed:.3f}")
        self.cmd_vel_pub.publish(cmd_vel)
    
    def stop_robot(self):
        """
        Stops the robot
        """
        msg = Twist()
        msg.linear.x = 0.0
        msg.angular.z = 0.0
        self.cmd_vel_pub.publish(msg)
    
    def inflate_obstacles(self, radius_cells=4):
        """
        Inflates occupied cells in the occupancy grid by a specified radius to create a safety buffer.
        Args:
            - radius_cells : number of grid cells to inflate around each occupied cell
        Returns:
            - inflated : 2D numpy array of the inflated occupancy grid
        """
        inflated = self.map_data.copy()
        occ = np.where(self.map_data == 100)  # occupied cells
        for y, x in zip(occ[0], occ[1]):
            for dy in range(-radius_cells, radius_cells + 1):
                for dx in range(-radius_cells, radius_cells + 1):
                    ny = y + dy
                    nx = x + dx
                    if 0 <= nx < self.map_width and 0 <= ny < self.map_height:
                        inflated[ny][nx] = 100
        
        return inflated
        
    def run(self):
        Frontier_selection = 0
        Path_follow = 1
        self.path_idx = 0
        state = Frontier_selection   # initial state
        path = None
        x_goal = y_goal = None
        goal_tolerance = 0.2 

        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05) 

            if not self.map_ready:
                self.get_logger().info("Waiting for map...")
                continue  
            
            # Replan path if map changed or no path exists
            if self.replan_needed or path is None:
                self.get_logger().info("Map updated → rebuilding graph + replanning")
                self.stop_robot()
                self.replan_needed = False

                # Rebuild graph
                self.map_processor = MapProcessor(self.map_data)
                self.map_processor.build_graph()

                if len(self.frontier_cells) == 0:
                    self.get_logger().info("Exploration complete.")
                    return

                # Select the closest frontier and plan
                x_goal, y_goal = self.closest_cell(self.frontier_cells)
                path = self.a_star_path_planner(self.x_robot, self.y_robot, x_goal, y_goal)
                self.path_idx = 0

                state = Path_follow # switch to path following

            elif state == Path_follow:
                # Check if path is valid
                if path is None or len(path.poses) == 0:
                    self.get_logger().warn("Path empty - re-selecting frontier.")
                    state = Frontier_selection
                    continue

                self.frontier_markers(self.frontier_cells)

                # Compute distance to final goal
                final_goal = path.poses[-1]
                dx = final_goal.pose.position.x - self.x_robot
                dy = final_goal.pose.position.y - self.y_robot
                dist_to_goal = math.sqrt(dx*dx + dy*dy)
                self.get_logger().info(f"Dist to goal:{dist_to_goal:.2f}")

                # Replan if map changed
                if self.replan_needed and dist_to_goal < goal_tolerance:
                    self.get_logger().info("Map changed near goal → replanning")
                    self.stop_robot()
                    self.replan_needed = False
                    state = Frontier_selection
                    continue
                
                # Update frontiers if closest frontier is reached
                if dist_to_goal < goal_tolerance:
                    self.get_logger().info("Reached frontier → updating frontiers.")
                    self.stop_robot()
                    self.centroids = []
                    self.goal_locked = False
                    state = Frontier_selection
                    continue
                
                # Follow the path toward current frontier
                self.path_idx = self.get_path_idx(path, self.ttbot_pose)
                current_goal = path.poses[self.path_idx]
                dist, heading_error, yaw_robot = self.path_follower(self.ttbot_pose, current_goal)
                self.move_ttbot(dist, heading_error)
        
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

class MapProcessor():
    def __init__(self,name):
        """
        Args:
            - name : base name of the map files
        Returns:
            - None
        """
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
    task1 = Task()
    try:
        task1.run()
    except KeyboardInterrupt:
        pass
    finally:
        task1.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()


