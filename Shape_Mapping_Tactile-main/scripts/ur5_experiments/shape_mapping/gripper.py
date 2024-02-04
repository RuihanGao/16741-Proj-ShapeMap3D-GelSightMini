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
		self.status = None

		rospy.Subscriber("/wsg_50_driver/status", Status, self.status_callback)

	def status_callback(self, status):
		# string status
		# time stamp
		# float32 width
		# float32 speed
		# float32 acc
		# float32 force
		# float32 force_finger0
		# float32 force_finger1
		self.status = status
		# print('gripper status: ', status)

	def get_status(self):
		# string status
		# time stamp
		# float32 width
		# float32 speed
		# float32 acc
		# float32 force
		# float32 force_finger0
		# float32 force_finger1
		print('gripper status: ', self.status)

	def no_contact(self):
		return

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


class GripperNode(object):
    def __init__(self):
        print("Initializing gripper node... ")

        rospy.init_node("gripper_node")

        # setup_srv()
        self.start_auto_update()
        rospy.spin()

    def start_auto_update(self):
        rospy.Subscriber("/wsg_50_driver/status", Status, self.ctrl_callback)

        self.pub = rospy.Publisher('/wsg_50_driver/goal_position', Cmd, queue_size=10)

