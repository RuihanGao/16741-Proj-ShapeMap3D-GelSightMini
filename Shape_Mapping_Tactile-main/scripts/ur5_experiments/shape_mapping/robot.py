#!/usr/bin/env python 

'''
Shubham Kanitkar (skanitka@andrew.cmu.edu) April, 2021
'''

import sys
import utils

import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg 
import numpy as np 

from record import start_data_collection, stop_data_collection
from scipy.spatial.transform import Rotation as R

## Global
moveit_commander.roscpp_initialize(sys.argv)
robot = moveit_commander.RobotCommander()
group_name = "manipulator"
move_group = moveit_commander.MoveGroupCommander(group_name)
display_trajectory = moveit_msgs.msg.DisplayTrajectory()
display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path',
												moveit_msgs.msg.DisplayTrajectory,
												queue_size=1)
vel_scale_factor = 0.2
acc_scale_factor = 0.1
planning_time = 10.0

class Robot():
	def __init__(self):
		self.robot = robot
		self.move_group = move_group
		self.display_trajectory = display_trajectory
		self.display_trajectory_publisher = display_trajectory_publisher

		self.move_group.set_planner_id('RRTstar')
		self.move_group.set_max_acceleration_scaling_factor(acc_scale_factor)
		self.move_group.set_max_velocity_scaling_factor(vel_scale_factor)
		self.move_group.set_planning_time(planning_time)

	# execute on y/n key input 
	def _user_input(self):
		key = raw_input()																# User input
		if key == 'y':
			self._execute_plan()
		elif key == 'n':
			print('Motion Declined!')
			print('\n')
		else:
			print('Invalid Input. Please select [y/n]')
			print('\n')
			self._user_input()

	def _execute_plan(self):
		self.move_group.execute(self.plan)											# Execute planned trajectory plan
		self.move_group.stop()														# Stop robot movement
		self.move_group.clear_pose_targets()										# Clear pose target before starting new planning
		print('Motion Completed!')
		print('\n')

	# plan trajectory with moveit planner, and check with user
	def _plan_trajectory(self, override = False):
		self.plan = self.move_group.plan()												# Plan the motion trajectory
		self.display_trajectory.trajectory_start = self.robot.get_current_state()		# Get robot current state
		self.display_trajectory.trajectory.append(self.plan)							# Schedule planned trajectory plan
		self.display_trajectory_publisher.publish(self.display_trajectory)				# Publish planned trajectory plan
		print('Publishing motion trajectory in RVIZ!')
		if override:
			print('Override mode: movement access automatic')
			self._execute_plan()
		else:
			print('Movement Access?? [y/n]: ')
			self._user_input()

	# set target in joint space
	def move_joint_space_goal(self, angle_base, angle_shoulder, angle_elbow, \
							angle_wrist1, angle_wrist2, angle_wrist3):

		joint_group_values = self.move_group.get_current_joint_values()					# Get current joint angle values
		joint_group_values[0] = angle_base												# Set new base joint value
		joint_group_values[1] = angle_shoulder											# Set new shoulder joint value
		joint_group_values[2] = angle_elbow												# Set new elbow joint value
		joint_group_values[3] = angle_wrist1											# Set new wrist1 joint value
		joint_group_values[4] = angle_wrist2											# Set new wrist2 joint value
		joint_group_values[5] = angle_wrist3											# Set new wrist3 joint value
		self.move_group.set_joint_value_target(joint_group_values)						# Set target joint values

		self._plan_trajectory()

	# offset current pose by (offset_x, offset_y, offset_z)
	def move_offset_distance(self, offset_x, offset_y, offset_z, override = False):

		pose_values = self.move_group.get_current_pose()
		pose_goal = geometry_msgs.msg.Pose()
		pose_goal.position = pose_values.pose.position
		pose_goal.orientation = pose_values.pose.orientation
		pose_goal.position.x += offset_x
		pose_goal.position.y += offset_y
		pose_goal.position.z += offset_z
		self.move_group.set_pose_target(pose_goal)
		self._plan_trajectory(override)

	# plan traj to (loc_x, loc_y, loc_z)
	def move_to_location(self, loc_x, loc_y, loc_z, override = False):

		pose_values = self.move_group.get_current_pose()
		pose_goal = geometry_msgs.msg.Pose()
		pose_goal.position = pose_values.pose.position
		pose_goal.orientation = pose_values.pose.orientation
		pose_goal.position.x = loc_x
		pose_goal.position.y = loc_y
		pose_goal.position.z = loc_z
		self.move_group.set_pose_target(pose_goal)
		self._plan_trajectory(override)

	# set target in joint space
	def rotate_wrist(self,  theta):
		joint_group_values = self.move_group.get_current_joint_values()					# Get current joint angle values
		joint_group_values[5] = theta											# Set new wrist3 joint value
		self.move_group.set_joint_value_target(joint_group_values)						# Set target joint values
		self._plan_trajectory()

	# plan traj to (loc_x, loc_y, loc_z)
	def rotate_to_angle(self, qw, qx, qy, qz):
		pose_values = self.move_group.get_current_pose()
		pose_goal = geometry_msgs.msg.Pose()
		pose_goal.position = pose_values.pose.position
		pose_goal.orientation.w = qw
		pose_goal.position.x = qx
		pose_goal.position.y = qy
		pose_goal.position.z = qz
		self.move_group.set_pose_target(pose_goal)
		self._plan_trajectory()

	# plan traj to (loc_x, loc_y, loc_z)
	def print_current_rotation(self):
		pose_values = self.move_group.get_current_pose()
		R_matrix = R.from_quat([pose_values.pose.orientation.x, pose_values.pose.orientation.y, pose_values.pose.orientation.z, pose_values.pose.orientation.w])
		eul = R_matrix.as_euler('zxy', degrees=True)
		print('End effector angles (degrees): ',  eul)

	# plan traj to (loc_x, loc_y, loc_z)
	def set_current_rotation(self, theta):
		pose_values = self.move_group.get_current_pose()

		current_R = R.from_quat([pose_values.pose.orientation.x, pose_values.pose.orientation.y, pose_values.pose.orientation.z, pose_values.pose.orientation.w])
		current_R = current_R.as_euler('zxy', degrees=False)
		new_R = R.from_euler('zxy', [current_R[0], theta + 1.57, current_R[2]], degrees=False)

		# print('New rotation: ', new_R.as_euler('zxy', degrees=True))
		quat = new_R.as_quat()
		pose_goal = geometry_msgs.msg.Pose()
		pose_goal.position = pose_values.pose.position
		pose_goal.orientation.w = quat[3]
		pose_goal.orientation.x = quat[0]
		pose_goal.orientation.y = quat[1]
		pose_goal.orientation.z = quat[2]
		self.move_group.set_pose_target(pose_goal)
		self._plan_trajectory()

	# plan traj to (loc_x, loc_y, loc_z)
	def move_to_pose(self, loc_x, loc_y, loc_z, qw, qx, qy, qz):

		pose_values = self.move_group.get_current_pose()
		pose_goal = geometry_msgs.msg.Pose()
		pose_goal.position = pose_values.pose.position
		pose_goal.orientation.w = qw
		pose_goal.position.x = qx
		pose_goal.position.y = qy
		pose_goal.position.z = qz
		pose_goal.position.x = loc_x
		pose_goal.position.y = loc_y
		pose_goal.position.z = loc_z
		self.move_group.set_pose_target(pose_goal)
		self._plan_trajectory()

	# Helper functions

	# Joint Angle Based Homing 
	def joint_angle_homing(self):
		print("Joint-Space Based Pre-grasp Position")
		self.move_joint_space_goal(angle_base = -1.57, angle_shoulder = -1.57, angle_elbow = -1.91, \
									angle_wrist1 = -1.22, angle_wrist2 = 1.57, angle_wrist3 = 1.57)

	def start_position(self):
		print("Joint-Space Based Pre-grasp Position")
		self.move_joint_space_goal(angle_base = np.radians(-98.81), angle_shoulder = np.radians(-104.07), angle_elbow = np.radians(-93.35), \
									angle_wrist1 = np.radians(-71.84), angle_wrist2 = np.radians(90.03), angle_wrist3 = np.radians(81.05))


	# Joint Angle Horizontal Homing 
	def joint_angle_horizontal_homing(self):
			print("Joint-Space Horizontal Pre-grasp Position")
			self.move_joint_space_goal(angle_base = -1.57, angle_shoulder = -1.57, angle_elbow = -1.91, \
										angle_wrist1 = -1.22, angle_wrist2 = 0, angle_wrist3 = 1.57)

	# teleoperate with (w, s, a, d, z, x, g, r)
	def manual_control(self, gripper):

		while True:
			direction = utils.robot_movement_direction_user_input()
			offset = float(raw_input('Enter Movement Distance: '))

			if direction == 'w':
				self.move_offset_distance(offset_x=0, offset_y=0, offset_z=offset)

			if direction == 's':
				self.move_offset_distance(offset_x=0, offset_y=0, offset_z=-offset)

			if direction == 'a':
				self.move_offset_distance(offset_x=offset, offset_y=0, offset_z=0)

			if direction == 'd':
				self.move_offset_distance(offset_x=-offset, offset_y=0, offset_z=0)

			if direction == 'z':
				self.move_offset_distance(offset_x=0, offset_y=offset, offset_z=0)

			if direction == 'x':
				self.move_offset_distance(offset_x=0, offset_y=-offset, offset_z=0)

			if direction == 'g':
				gripper.graspWithForce()

			if direction == 'r':
				gripper.homing()

			if (utils.reached_destination()):
				break