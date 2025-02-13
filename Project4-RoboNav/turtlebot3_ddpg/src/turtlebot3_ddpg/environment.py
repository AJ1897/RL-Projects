#!/usr/bin/env python
#################################################################################
# Copyright 2018 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#################################################################################

# Authors: Gilbert #

import rospy
import numpy as np
import math
from math import pi
from geometry_msgs.msg import Twist, Point, Pose
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import sys
import os
# sys.path.append('/home/aj18/catkin_ws/src/turtlebot3_dqn/src/turtlebot3_dqn/')
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from respawnGoal import Respawn

class Env():
    def __init__(self, action_size):
        self.goal_x = 0
        self.goal_y = 0
        self.heading = 0
        self.action_size = action_size
        self.initGoal = True
        self.get_goalbox = False
        self.goal_reached = False
        self.position = Pose()
        self.pub_cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=5)
        self.sub_odom = rospy.Subscriber('odom', Odometry, self.getOdometry)
        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty)
        self.unpause_proxy = rospy.ServiceProxy('gazebo/unpause_physics', Empty)
        self.pause_proxy = rospy.ServiceProxy('gazebo/pause_physics', Empty)
        self.respawn_goal = Respawn()
        self.stage = rospy.get_param('stage_number')

    def getGoalDistace(self):
        goal_distance = round(math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y), 2)

        return goal_distance

    def getOdometry(self, odom):
        self.position = odom.pose.pose.position
        orientation = odom.pose.pose.orientation
        orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w]
        _, _, yaw = euler_from_quaternion(orientation_list)

        goal_angle = math.atan2(self.goal_y - self.position.y, self.goal_x - self.position.x)

        heading = goal_angle - yaw
        if heading > pi:
            heading -= 2 * pi

        elif heading < -pi:
            heading += 2 * pi

        self.heading = round(heading, 2)

    def getState(self, scan):
        scan_range = []
        heading = self.heading
        min_range = 0.13
        done = False

        for i in range(len(scan.ranges)):
            if scan.ranges[i] == float('Inf'):
                scan_range.append(3.5)
            elif np.isnan(scan.ranges[i]):
                scan_range.append(0)
            else:
                scan_range.append(scan.ranges[i])

        obstacle_min_range = round(min(scan_range), 2)
        obstacle_angle = np.argmin(scan_range)
        if min_range > min(scan_range) > 0:
            done = True

        current_distance = round(math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y),2)
        if current_distance < 0.2:
            self.get_goalbox = True
            self.goal_reached = True

        if self.stage == 1:
            return scan_range + [heading, current_distance], done
        else:
            return scan_range + [heading, current_distance, obstacle_min_range, obstacle_angle], done

    def setReward(self, state, done, action):
        yaw_reward = []
        if self.stage == 1:
            current_distance = state[-1]
            heading = state[-2]
        else:
            obstacle_min_range = state[-2]
            current_distance = state[-3]
            heading = state[-4]
        additional_term = -pi/4*(action[0]-1)

        angle = -pi / 4 + heading + additional_term + pi / 2
        tr = 1 - 4 * math.fabs(0.5 - math.modf(0.25 + 0.5 * angle % (2 * math.pi) / math.pi)[0])
        yaw_reward = tr

        distance_rate = 2 ** (current_distance / self.goal_distance) if self.goal_distance > 0 else 100

        if self.stage == 4:
            if obstacle_min_range < 0.5:
                ob_reward = -5
            else:
                ob_reward = 0
            reward = ((round(yaw_reward * 5, 2)) * distance_rate) + ob_reward
        else:
            reward = ((round(yaw_reward * 5, 2)) * distance_rate)

        if done:
            rospy.loginfo("Collision!!")
            if self.stage == 1:
                reward = -200
            elif self.stage == 4:
                reward = -500
            else:
                reward = -150
            self.pub_cmd_vel.publish(Twist())

        if self.get_goalbox:
            rospy.loginfo("Goal!!")
            if self.stage == 4:
                reward = 1000
            else:
                reward = 200
            self.pub_cmd_vel.publish(Twist())
            goal_count = rospy.get_param('goal_count')
            if goal_count==4:
                goal_count += 1
                rospy.set_param('goal_count', goal_count)
            if goal_count < 4:
                goal_count += 1
                rospy.set_param('goal_count', goal_count)
                self.goal_x, self.goal_y = self.respawn_goal.getPosition(True, delete=True)
                self.goal_distance = self.getGoalDistace()
            self.get_goalbox = False

        return reward


    def step(self, action):
        #max_angular_vel = 1.5
        #ang_vel = ((self.action_size - 1)/2 - action) * max_angular_vel * 0.5
        max_angular_vel = 2
        min_angular_vel = -2
        sigma = (max_angular_vel - min_angular_vel) / 2
        mean = (max_angular_vel + min_angular_vel) / 2
        ang_vel = sigma * action[0] + mean

        # max_linear_velocity = 0.5
        # min_linear_velocity = 0

        # sigma_l = (max_linear_velocity - min_linear_velocity) / 2
        # mean_l = (max_linear_velocity + min_linear_velocity) / 2
        # linear_velocity = sigma_l * action[1] + mean_l
        linear_velocity = 0.15
        
        vel_cmd = Twist()
        vel_cmd.linear.x = linear_velocity
        vel_cmd.angular.z = ang_vel
        self.pub_cmd_vel.publish(vel_cmd)

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('scan', LaserScan, timeout=5)
            except:
                pass

        state, done = self.getState(data)
        reward = self.setReward(state, done, action)

        return np.asarray(state), reward, done

    def reset(self):
        rospy.wait_for_service('gazebo/reset_simulation')
        try:
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print("gazebo/reset_simulation service call failed")

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('scan', LaserScan, timeout=5)
            except:
                pass

        if self.initGoal:
            self.goal_x, self.goal_y = self.respawn_goal.getPosition()
            self.initGoal = False

        self.goal_distance = self.getGoalDistace()
        state, done = self.getState(data)

        return np.asarray(state)
