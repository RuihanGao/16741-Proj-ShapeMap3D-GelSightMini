#!/usr/bin/env python 

'''
Shape mapping of objects with GelSight, data collection script 
Zilin Si, Sudharshan Suresh (zsi@andrew.cmu.edu, suddhu@cmu.edu)
Based on original scripts by Shubham Kanitkar (skanitka@andrew.cmu.edu / shubhamkanitkar32@gmail.com) 
July 2021
'''


import rospy
import moveit_commander
import moveit_msgs.msg
import tf
import os
import csv

from robot import Robot
from gripper import Gripper
from algo import explore_object, explore_object_rect
from utils import load_object_properties, create_data_dir
from record import collect_frame, start_data_collection, stop_data_collection

# Global Variables
parent_dir = '/media/okemo/extraHDD1/Zilin/shape_mapping'									# Data collection location
log_dir = parent_dir + 'log.csv'															# log.csv to keep track of the data collection

# Main function
def main():
	rospy.init_node('main', anonymous=True)													# ROS init node
	rate = rospy.Rate(100)																	# Set ROS rate

	robot = Robot()																			# Robot object
	gripper = Gripper()																		# Gripper object

	##### test contact stuff 
	# trial_dir, _ = create_data_dir(parent_dir, "debug")					    		# Create directory for data collection
	# azure, tactile, usbcam, ur5e = start_data_collection(trial_dir)	# Start data collection
	# tactile.saveBackground()
	# while True:
	# 	while (not tactile.detectContact()):
	# 		pass

	# 	print("Contact detected!")
	# 	tactile.debugContact()
	# stop_data_collection(azure=azure, tactile=tactile, wsg50=None, usbcam=usbcam, ur5e=ur5e)				# Stop data collection
	# return	
	##### test contact stuff 

	
	object = load_object_properties(hardcode = False)										# Load properties for object
	trial_dir, _ = create_data_dir(parent_dir, object["name"])					    		# Create directory for data collection

	while not rospy.is_shutdown():
		rate.sleep()
		try:
			
			print('START: Robot/gripper homing')
			robot.joint_angle_homing()														# Robot joints homing (move to preset location)
			gripper.homing()																# Gripper homing (maximum opening)
			
			if object["name"] == "002_master_chef_can" or object["name"] == "005_tomato_soup_can" or object["name"] == "021_bleach_cleanser":
				explore_object(robot, gripper, object, trial_dir) 					            # single command to perform the exploration for an object 
			else: 
				explore_object_rect(robot, gripper, object, trial_dir) 
			
			print('END: Robot/gripper homing')
			robot.joint_angle_homing()														# Robot joints homing (move to preset location)
			gripper.homing()	
			break

		except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
			print('FATAL: Error in robot control')
			continue

if __name__ == '__main__':
	main()