#!/usr/bin/env python 

'''
Shubham Kanitkar (skanitka@andrew.cmu.edu) April, 2021
'''

import rospy
from wsg_50_common.srv import *
from wsg_50_common.msg import Status
from std_srvs.srv import Empty


class Gripper():
	def __init__(self, width=0, speed=50.0, force=80):
		self.width = width
		self.speed = speed
		self.force = force

	# Grasps an object of a specific width at a specific velocity
	def grasp(self):
		rospy.wait_for_service('/wsg_50_driver/grasp')
		try:
			service = rospy.ServiceProxy('/wsg_50_driver/grasp', Move)
			service(self.width, self.speed)
		except rospy.ServiceException, e:
			print ("Service Call Failed: %s"%e)

	# Moves fingers to home position (maximum opening)
	def homing(self):
		rospy.wait_for_service('/wsg_50_driver/homing')
		try:

			service = rospy.ServiceProxy('/wsg_50_driver/homing', Empty)
			service()
		except rospy.ServiceException as e:
			print("Service call failed: %s"%e)

	# Moves fingers to an absolute position at a specific velocity
	def move(self):
		rospy.wait_for_service('/wsg_50_driver/homing')
		try:
			service = rospy.ServiceProxy('/wsg_50_driver/move', Move)
			service(self.width, self.speed)
		except rospy.ServiceException, e:
			print("Service call failed: %s"%e)

	# Releases a grasped object opening the fingers to a indicated position
	def release(self):
		rospy.wait_for_service('/wsg_50_driver/release')
		try:
			service = rospy.ServiceProxy('/wsg_50_driver/release', Move)
			service(self.width, self.speed)
		except rospy.ServiceException, e:
			print("Service call failed: %s"%e)

	# Set the force with the gripper grasp objects
	def graspWithForce(self):	
		rospy.wait_for_service('/wsg_50_driver/graspForce')
		try:
			_gp = rospy.ServiceProxy('/wsg_50_driver/graspForce', GraspForce)
			resp = _gp(self.force, self.speed)
			self.grasp()
			return resp.error
		except rospy.ServiceException as e:
			print("Service call failed %s"%e) 

