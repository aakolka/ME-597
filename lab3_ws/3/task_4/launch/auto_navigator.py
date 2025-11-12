#!/usr/bin/env python3
from PIL import Image, ImageOps
from ament_index_python.packages import get_package_share_directory
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import sys
import os
import yaml
from copy import copy, deepcopy
import time
import math
import heapq
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Pose, Twist
from std_msgs.msg import Float32

class Navigation(Node):
    """! Navigation node class.
    This class should serve as a template to implement the path planning and
    path follower components to move the turtlebot from position A to B.
    """

    def __init__(self, map_processor,node_name='Navigation'):
        """! Class constructor.
        @param  None.
        @return An instance of the Navigation class.
        """
        super().__init__(node_name)
        # Path planner/follower related variables
        self.path = Path()
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
        """! Callback to catch the goal pose.
        @param  data    PoseStamped object from RVIZ.
        @return None.
        """
        self.goal_pose = data
        self.get_logger().info(
            'goal_pose: {:.4f}, {:.4f}'.format(self.goal_pose.pose.position.x, self.goal_pose.pose.position.y))
    
    def goal_callback(self, msg):
        self.goal_pose = msg
    
    def __ttbot_pose_cbk(self, data):
        """! Callback to catch the position of the vehicle.
        @param  data    PoseWithCovarianceStamped object from amcl.
        @return None.
        """
        self.ttbot_pose = data.pose
        self.get_logger().info(
            'ttbot_pose: {:.4f}, {:.4f}'.format(self.ttbot_pose.pose.position.x, self.ttbot_pose.pose.position.y))

    def pose_to_grid(self, vehicle_pose):
        x_world = vehicle_pose.pose.position.x
        y_world = vehicle_pose.pose.position.y
        
        j = int((x_world - self.x_min) / self.resolution)  
        i = int(self.map_height - 1 - (y_world - self.y_min) / self.resolution)
    
        return i, j  
    
    def grid_to_pose(self,i, j):
        x_world = self.x_min + (j + 0.5) * self.resolution
        y_world = self.y_min + (self.map_height -1 - i + 0.5) * self.resolution
    
        return x_world, y_world

    def a_star_path_planner(self, start_pose, end_pose):
        """! A Start path planner.
        @param  start_pose    PoseStamped object containing the start of the path to be created.
        @param  end_pose      PoseStamped object containing the end of the path to be created.
        @return path          Path object containing the sequence of waypoints of the created path.
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

        astar = AStar(self.map_processor.map_graph)
        astar.solve(start_node,end_node)
        path_grid,dist = astar.reconstruct_path(start_node,end_node)
        # Draw the path on the map for visualization/debugging
        self.get_logger().info('A* found path with %d waypoints, distance: %.2f' % (len(path_grid), dist))
        #self.map_processor.path_coords(path_grid)
        #self.map_processor.draw_path_graph(path_grid)

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
        """! Path follower.
        @param  path                  Path object containing the sequence of waypoints of the created path.
        @param  current_goal_pose     PoseStamped object containing the current vehicle position.
        @return idx                   Position in the path pointing to the next goal pose to follow.
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


        if dist_to_final_goal > goal_threshold:
            lookahead_distance = max_lookahead
        else:
            lookahead_distance = min_lookahead + (max_lookahead - min_lookahead) * (dist_to_final_goal / goal_threshold)
        
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
        """! Path follower.
        @param  vehicle_pose           PoseStamped object containing the current vehicle pose.
        @param  current_goal_pose      PoseStamped object containing the current target from the created path. This is different from the global target.
        @return path                   Path object containing the sequence of waypoints of the created path.
        """
        #speed = 0.0
        #heading = 0.0
        # TODO: IMPLEMENT PATH FOLLOWER
        x_robot = vehicle_pose.pose.position.x
        y_robot = vehicle_pose.pose.position.y
        x_goal = current_goal_pose.pose.position.x
        y_goal = current_goal_pose.pose.position.y

        dist = math.sqrt((x_robot - x_goal) ** 2 + (y_robot - y_goal) ** 2)
        desired_heading = math.atan2((y_goal - y_robot),(x_goal - x_robot))

        qx = vehicle_pose.pose.orientation.x
        qy = vehicle_pose.pose.orientation.y
        qz = vehicle_pose.pose.orientation.z
        qw = vehicle_pose.pose.orientation.w
        yaw_robot = math.atan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))
        
        heading_error = desired_heading - yaw_robot
        heading_error = math.atan2(math.sin(heading_error), math.cos(heading_error))

        #self.get_logger().info(f"Robot pose: ({x_robot:.3f}, {y_robot:.3f}) | Goal pose: ({x_goal:.3f}, {y_goal:.3f})")
        self.get_logger().info(f"x_goal={x_goal:.4f}, y_goal={y_goal:.4f}, x_robot={x_robot:.4f}, y_robot={y_robot:.4f}, desired_heading={math.degrees(desired_heading):.4f}, \
                               yaw_robot={math.degrees(yaw_robot):.4f}, heading_error={math.degrees(heading_error):.4f}")

        return dist, heading_error, yaw_robot

    def move_ttbot(self, distance, heading_error):
        """! Function to move turtlebot passing directly a heading angle and the speed.
        @param  speed     Desired speed.
        @param  heading   Desired yaw angle.
        @return path      object containing the sequence of waypoints of the created path.
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

        linear_speed = (
            kp_lin * lin_error +
            ki_lin * self.integral_lin * 0 +
            kd_lin * derivative_lin
        )
        linear_speed = max(min(linear_speed, max_speed), -max_speed)

        ang_error = heading_error
        self.integral_ang += ang_error * dt
        derivative_ang = (ang_error - self.prev_ang_error) / dt

        angular_speed = (
            kp_ang * ang_error +
            ki_ang * self.integral_ang * 0+
            kd_ang * derivative_ang 
        )
        angular_speed = max(min(angular_speed, max_ang_speed), -max_ang_speed)
        
        self.prev_lin_error = lin_error
        self.prev_ang_error = ang_error

        heading_threshold = math.radians(30)  # ~30 degrees
        if abs(heading_error) > heading_threshold:
            self.get_logger().info(f"[TURN-IN-PLACE] Large heading error ({math.degrees(heading_error):.2f}°) → rotating only")
            linear_speed = 0.0  # stop forward motion

        # Stop rotating if nearly aligned
        #if abs(heading_error) > math.radians(15):
        #    linear_speed = 0.0  # stop forward motion
        #else:
        #    linear_speed = max(min(linear_speed, max_speed), -max_speed)

        # If almost aligned, reduce angular speed noise
        #if abs(heading_error) < math.radians(3):
        #    angular_speed = 0.0
       
        cmd_vel = Twist()
        cmd_vel.linear.x = linear_speed
        cmd_vel.angular.z = angular_speed

        self.get_logger().info(f"[MOVE] Linear Speed: {linear_speed:.3f}, Angular Speed: {angular_speed:.3f}")
        self.cmd_vel_pub.publish(cmd_vel)

    def run(self):
        path = None
        self.replan_needed = True
        self.path_idx = 0
        goal_tolerance = 0.1  # meters
        
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)

            if (self.ttbot_pose.pose.position.x == 0.0 and self.ttbot_pose.pose.position.y == 0.0) or \
            (self.goal_pose.pose.position.x == 0.0 and self.goal_pose.pose.position.y == 0.0):
                continue  

            # Plan only once
            if self.replan_needed:
                path = self.a_star_path_planner(self.ttbot_pose, self.goal_pose)
                self.replan_needed = False
                self.path_idx = 0
                        
            self.path_idx = self.get_path_idx(path, self.ttbot_pose)
            current_goal = path.poses[self.path_idx]
            dist, heading_error,yaw_robot = self.path_follower(self.ttbot_pose, current_goal)

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

            self.move_ttbot(dist, heading_error)
            self.get_logger().info("[RUN] Tick loop alive") #del
            #self.rate.sleep()

class Queue():
    def __init__(self, init_queue = []):
        self.queue = copy(init_queue)
        self.start = 0
        self.end = len(self.queue)-1

    def __len__(self):
        numel = len(self.queue)
        return numel

    def __repr__(self):
        q = self.queue
        tmpstr = ""
        for i in range(len(self.queue)):
            flag = False
            if(i == self.start):
                tmpstr += "<"
                flag = True
            if(i == self.end):
                tmpstr += ">"
                flag = True

            if(flag):
                tmpstr += '| ' + str(q[i]) + '|\n'
            else:
                tmpstr += ' | ' + str(q[i]) + '|\n'

        return tmpstr

    def __call__(self):
        return self.queue

    def initialize_queue(self,init_queue = []):
        self.queue = copy(init_queue)

    def sort(self,key=str.lower):
        self.queue = sorted(self.queue,key=key)

    def push(self,data):
        self.queue.append(data)
        self.end += 1

    def pop(self):
        p = self.queue.pop(self.start)
        self.end = len(self.queue)-1
        return p


class AStar():
    def __init__(self,in_tree):
        self.in_tree = in_tree
        self.q = Queue()
        self.dist = {name:np.inf for name,node in in_tree.g.items()}
        self.h = {name:0 for name,node in in_tree.g.items()}

        for name,node in in_tree.g.items():
            start = tuple(map(int, name.split(',')))
            end = tuple(map(int, self.in_tree.end.split(',')))
            self.h[name] = np.sqrt((end[0]-start[0])**2 + (end[1]-start[1])**2)

        self.via = {name:0 for name,node in in_tree.g.items()}
        for __,node in in_tree.g.items():
            self.q.push(node)

    def __get_f_score(self,node):
        return self.dist[node.name] + self.h[node.name]

    def solve(self, sn, en):
        '''self.dist[sn.name] = 0
        while len(self.q) > 0:
          self.q.sort(key = self.__get_f_score)
          u = self.q.pop()
          if u.name == en.name:
            break
          for i in range(len(u.children)):
            c = u.children[i]
            w = u.weight[i]
            new_dist = self.dist[u.name] + w
            if new_dist < self.dist[c.name]:
              self.dist[c.name] = new_dist
              self.via[c.name] = u.name'''
        
        self.dist[sn.name] = 0
        open_set = [sn]
        closed_set = set()

        while open_set:
            # Pick the node with the lowest f-score
            u = min(open_set, key=self.__get_f_score)
            open_set.remove(u)

            if u.name == en.name:
                break

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
        end_key = en.name
        u = end_key
        path = [u]
        dist = self.dist[end_key]
        start_key = sn.name
        while u!=start_key:
          u = self.via[u]
          path.append(u)
        path.reverse()
        return path,dist


class Node():
    def __init__(self,name):
        self.name = name
        self.children = []
        self.weight = []

    def __repr__(self):
        return self.name

    def add_children(self,node,w=None):
        if w == None:
            w = [1]*len(node)
        self.children.extend(node)
        self.weight.extend(w)


class Tree():
    def __init__(self,name):
        self.name = name
        self.root = 0
        self.end = 0
        self.g = {}


    def add_node(self, node, start = False, end = False):
        self.g[node.name] = node
        if(start):
            self.root = node.name
        elif(end):
            self.end = node.name

    def set_as_root(self,node):
        # These are exclusive conditions
        self.root = True
        self.end = False

    def set_as_end(self,node):
        # These are exclusive conditions
        self.root = False
        self.end = True



class Map():
    def __init__(self, map_name):
        self.map_im, self.map_df, self.limits = self.__open_map(map_name)
        self.image_array = self.__get_obstacle_map(self.map_im, self.map_df)

    def __repr__(self):
        fig, ax = plt.subplots(dpi=150)
        ax.imshow(self.image_array,extent=self.limits, cmap=cm.gray)
        ax.plot()
        return ""

    def __open_map(self,map_name):
        # Open the YAML file which contains the map name and other
        # configuration parameters
        f = open(map_name + '.yaml', 'r')
        map_df = pd.json_normalize(yaml.safe_load(f))
        # Open the map image
        map_name = map_df.image[0]
        im = Image.open(map_name)
        #size = 200, 200
        #im.thumbnail(size)
        im = ImageOps.grayscale(im)
        # Get the limits of the map. This will help to display the map
        # with the correct axis ticks.
        xmin = map_df.origin[0][0]
        xmax = map_df.origin[0][0] + im.size[0] * 0+ map_df.resolution[0]
        ymin = map_df.origin[0][1]
        ymax = map_df.origin[0][1] + im.size[1] * 0+  map_df.resolution[0]

        return im, map_df, [xmin,xmax,ymin,ymax]

    def __get_obstacle_map(self,map_im, map_df):
        img_array = np.reshape(list(self.map_im.getdata()),(self.map_im.size[1],self.map_im.size[0]))
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
        self.map = Map(name)
        self.inf_map_img_array = np.zeros(self.map.image_array.shape)
        self.map_graph = Tree(name)

    def __modify_map_pixel(self,map_array,i,j,value,absolute):
        if( (i >= 0) and
            (i < map_array.shape[0]) and
            (j >= 0) and
            (j < map_array.shape[1]) ):
            if absolute:
                map_array[i][j] = value
            else:
                map_array[i][j] += value

    def __inflate_obstacle(self,kernel,map_array,i,j,absolute):
        dx = int(kernel.shape[0]//2)
        dy = int(kernel.shape[1]//2)
        if (dx == 0) and (dy == 0):
            self.__modify_map_pixel(map_array,i,j,kernel[0][0],absolute)
        else:
            for k in range(i-dx,i+dx):
                for l in range(j-dy,j+dy):
                    self.__modify_map_pixel(map_array,k,l,kernel[k-i+dx][l-j+dy],absolute)

    def inflate_map(self,kernel,absolute=True):
        # Perform an operation like dilation, such that the small wall found during the mapping process
        # are increased in size, thus forcing a safer path.
        self.inf_map_img_array = np.zeros(self.map.image_array.shape)
        for i in range(self.map.image_array.shape[0]):
            for j in range(self.map.image_array.shape[1]):
                if self.map.image_array[i][j] == 0:
                    self.__inflate_obstacle(kernel,self.inf_map_img_array,i,j,absolute)
        r = np.max(self.inf_map_img_array)-np.min(self.inf_map_img_array)
        if r == 0:
            r = 1
        self.inf_map_img_array = (self.inf_map_img_array - np.min(self.inf_map_img_array))/r

    def get_graph_from_map(self):
        # Create the nodes that will be part of the graph, considering only valid nodes or the free space
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
        size = int(size) // 2
        x, y = np.mgrid[-size:size+1, -size:size+1]
        normal = 1 / (2.0 * np.pi * sigma**2)
        g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
        r = np.max(g)-np.min(g)
        sm = (g - np.min(g))*1/r
        return sm

    def rect_kernel(self, size, value):
        m = np.ones(shape=(size,size))
        return m

    def path_coords(self, path):
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
    mp = MapProcessor("/home/me597/sim_ws/src/turtlebot3_gazebo/maps/classroom_map")
    kernel = mp.rect_kernel(12, 1)   # or mp.gaussian_kernel(3, sigma=1)
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