#!/usr/bin/env python3

import rclpy
import math
import rclpy.time
from rclpy.node import Node
from example_interfaces.msg import String
import numpy as np

from geometry_msgs.msg import Twist, Pose
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
import time
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from geometry_msgs.msg import PoseStamped, Pose


def quaternion_to_euler(x, y, z, w):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    """
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)  # use 90 degrees if out of range
    else:
        pitch = math.asin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw

class Publisher(Node):
    def __init__(self):
        super().__init__("publisher")
        self.pose = Pose()
        self.frame = 'map'
        self.navigator = BasicNavigator()
        self.i = 0
        self.yaw = 0.0
        self.initial_pose = (-2.0, 2.0)
        self.robotPos = self.initial_pose
        self.current_pos = self.initial_pose
        
        self.targets = [
            (2.2, 2.2),    # Green Block
            (2.15, -2.15), # Red Block  
            (-2.16, -2.16), # Blue Block
            (0.09, 0.6),     # Middle Block
            (-2.0, 1.2),    # Orange Block
        ]
        self.target_names = ["Green Block", "Red Block", "Blue Block", "Middle Block", "Orange Block"]

        # Mission state management
        self.current_target_index = 0
        self.mission_state = "GO_TO_TARGET"  # "GO_TO_TARGET" or "RETURN_TO_ORIGIN"
        self.mission_completed = False
        self.picking_up_item = False
        
        # Laser zones for obstacle avoidance
        self.zone_F = [0.0]
        self.zone_FL = [0.0]
        self.zone_L = [0.0]
        self.zone_BL = [0.0]
        self.zone_B = [0.0]
        self.zone_BR = [0.0]
        self.zone_R = [0.0]
        self.zone_FR = [0.0]

        # ROS subscriptions and publishers
        self.odom_subscription = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.laserScan_subscription = self.create_subscription(LaserScan, "/scan", self.scan_callback, 10)
        self.publisher = self.create_publisher(Twist, "/cmd_vel", 10)

        self.get_logger().info("TurtleBot3 Mission Controller Initialized")
        self.get_logger().info(f"Starting position: {self.initial_pose}")
        self.get_logger().info(f"First target: {self.target_names[0]} at {self.targets[0]}")
        
        # Rotation matrix for coordinate transformation (as mentioned in PDF)
        self.rotation_matrix_initialized = False

    def scan_callback(self, data: LaserScan):
        # Initialize rotation matrix and position reference (as mentioned in PDF)
        if not self.rotation_matrix_initialized:
            self.rotation_matrix_initialized = True
            # self.get_logger().info("Rotation matrix initialized for coordinate transformation")
        
        self.maxRange = data.range_max
        self.minRange = data.range_min
        zone = np.array(data.ranges)
        
        self.zone_F = np.concatenate((zone[0:45], zone[315:360]))  # Front zone
        self.zone_FL = zone[26:67]    # Front-left
        self.zone_L = zone[46:134]    # Left
        self.zone_BL = zone[114:159]  # Back-left
        self.zone_B = zone[135:225]   # Back
        self.zone_BR = zone[206:251]  # Back-right
        self.zone_R = zone[226:314]   # Right
        self.zone_FR = zone[298:337]  # Front-right
        
        # Enhanced front zone with cosine correction for better obstacle detection
        self.zone_FRENTE = np.concatenate((zone[0:90], np.flip(zone[270:360])))
        for index, dist in enumerate(self.zone_FRENTE):
            if dist > 0:
                self.zone_FRENTE[index] = dist / abs(math.cos(math.radians(index)))

        if self.mission_completed:
            self.stop_robot()
            return

        if self.mission_state == "GO_TO_TARGET":
            if self.current_target_index < len(self.targets):
                target_pos = self.targets[self.current_target_index]
                target_name = self.target_names[self.current_target_index]
                
                if self.navigate_to_position(target_pos, target_name):
                    self.get_logger().warn(f"Reached {target_name}!")

                    if target_name == 'Blue Block':
                        self.current_target_index += 1
                        next_target = self.target_names[self.current_target_index]
                        self.mission_state = "GO_TO_TARGET"
                    else:
                        self.mission_state = "RETURN_TO_ORIGIN"
                    
        elif self.mission_state == "RETURN_TO_ORIGIN":
            if self.navigate_to_position(self.initial_pose, "Origin"):
                self.get_logger().warn(f"Returned to origin! Mission {self.current_target_index + 1}/4 completed!")
                self.current_target_index += 1
                
                if self.current_target_index >= len(self.targets):
                    self.mission_completed = True
                    self.get_logger().warn("ALL MISSIONS COMPLETED! Robot has visited all blocks and returned to origin!")
                    self.stop_robot()
                else:
                    next_target = self.target_names[self.current_target_index]
                    self.get_logger().warn(f"Next mission: {next_target} at {self.targets[self.current_target_index]}")
                    self.mission_state = "GO_TO_TARGET"

    def odom_callback(self, msg: Odometry):
        robotPos_x = msg.pose.pose.position.x
        robotPos_y = msg.pose.pose.position.y

        self.robotPos = robotPos_x, robotPos_y

        (_, _, self.yaw) = quaternion_to_euler(
            msg.pose.pose.orientation.x, 
            msg.pose.pose.orientation.y, 
            msg.pose.pose.orientation.z, 
            msg.pose.pose.orientation.w
        )

    def navigate_to_position(self, target_pos, target_name):
        """Navigate to a target position with obstacle avoidance"""
        cmd_vel_msg = Twist()
        
        distance_x = target_pos[0] - self.robotPos[0]
        distance_y = target_pos[1] - self.robotPos[1]
        distance = math.sqrt(distance_x ** 2 + distance_y ** 2)
        theta = math.atan2(distance_y, distance_x)
        
        # Check if we reached the target (tolerance of 0.3 meters as specified in PDF)
        if abs(distance_x) < 0.3 and abs(distance_y) < 0.3:
            self.get_logger().warn(f"Reached {target_name}! Position: ({self.robotPos[0]:.2f}, {self.robotPos[1]:.2f})")
            cmd_vel_msg.linear.x = 0.0
            cmd_vel_msg.angular.z = 0.0
            self.publisher.publish(cmd_vel_msg)
            return True
        
        # Obstacle avoidance logic
        front_distance = min(self.zone_F)
        left_distance = min(self.zone_L)
        right_distance = min(self.zone_R)
        
        # Angular difference to target
        angle_diff = theta - self.yaw
        # Normalize angle to [-pi, pi]
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi
        
        if front_distance < 0.3:
            # Obstacle in front - turn towards clearer side
            cmd_vel_msg.linear.x = 0.0
            if left_distance > right_distance:
                cmd_vel_msg.angular.z = 0.4  # Turn left
            else:
                cmd_vel_msg.angular.z = -0.4  # Turn right
        elif left_distance < 0.25:
            # Obstacle on left - adjust slightly right
            cmd_vel_msg.linear.x = 0.2
            cmd_vel_msg.angular.z = -0.3
        elif right_distance < 0.25:
            # Obstacle on right - adjust slightly left
            cmd_vel_msg.linear.x = 0.2
            cmd_vel_msg.angular.z = 0.3
        else:
            # Clear path - navigate towards target
            # First align with target direction
            if abs(angle_diff) > 0.3:
                cmd_vel_msg.linear.x = 0.2
                cmd_vel_msg.angular.z = 0.5 * angle_diff
            else:
                # Move towards target
                speed = 0.5  # Adaptive speed based on distance
                cmd_vel_msg.linear.x = speed
                cmd_vel_msg.angular.z = 0.5 * angle_diff
        
        self.publisher.publish(cmd_vel_msg)
        
        return False

    def stop_robot(self):
        """Stop the robot completely"""
        cmd_vel_msg = Twist()
        cmd_vel_msg.linear.x = 0.0
        cmd_vel_msg.angular.z = 0.0
        self.publisher.publish(cmd_vel_msg)
def main(args=None):
    rclpy.init(args=None)
    node = Publisher()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()
