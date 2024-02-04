#!/usr/bin/env python 

'''
Main function for robot manipulation and data collection
Written by Shubham Kanitkar (skanitka@andrew.cmu.edu / shubhamkanitkar32@gmail.com) 
Last revision: June 2021
'''


import rospy
import moveit_commander
import moveit_msgs.msg
import tf
import os
import csv

from robot import Robot
from gripper import Gripper
from record import start_data_collection, stop_data_collection
from algo import movement_cycle
import utils


# Global Variables
table_center = [0, 0.55, 0.1]																# Object location on the table
parent_dir = '/media/okemo/extraHDD1/Okemo/data'											# Data collection location
log_dir = parent_dir + 'log.csv'															# log.csv to keep track of the data collection


# Main function
def main():
	rospy.init_node('main', anonymous=True)													# ROS init node
	rate = rospy.Rate(100)																	# Set ROS rate

	robot = Robot()																			# Robot object
	gripper = Gripper()																		# Gripper object

	trial_dir, trial_path,= utils.create_data_dir(parent_dir)								# Create directory for data collection

	while not rospy.is_shutdown():
		rate.sleep()
		try:
			robot.joint_angle_homing()														# Robot joints homing
			gripper.homing()																# Gripper homing

			# Move robot end-effector to table center
			robot.move_to_location(	loc_x=table_center[0]-0.025, loc_y=table_center[1], loc_z=table_center[2]+0.2)

			# Decrease z-distance to reach the object
			robot.move_offset_distance(	offset_x=0, offset_y=0, offset_z=-0.09)

			azure, tactile, wsg50, usbcam, onrobot, ur5e = start_data_collection(trial_dir)	# Start data collection
			movement_cycle(trial_dir, log_dir, robot, gripper)								# Perform algorithm part
			stop_data_collection(azure, tactile, wsg50, usbcam, onrobot, ur5e)				# Stop data collection

			print('Done!')
			break

		except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
			continue

if __name__ == '__main__':
	main()