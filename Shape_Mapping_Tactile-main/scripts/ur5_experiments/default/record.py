#!/usr/bin/env python 

'''
Shubham Kanitkar (skanitka@andrew.cmu.edu) April, 2021
'''
import os
import csv
import cv2
import rospy
import time
import utils
import numpy as np

from skimage.io import imsave
from cv_bridge import CvBridge, CvBridgeError
from datetime import datetime

from sensor_msgs.msg import Image, JointState, CompressedImage
from geometry_msgs.msg import Wrench


class Sensor:
	def __init__(self, object_dir):
		self.object_dir = object_dir

	def __str__(self):
		return 'General'


'''
Sensor: Azure Kinect [Side View]
Data: RGB Compressed and RGB-D Raw
Format: .jpg
'''
class Azure(Sensor):
	def __init__(self, object_dir):
		self.object_dir = object_dir
		self.bridge = CvBridge()

		# RGB
		self.rgb_count = 0
		rgb_directory = 'rgb'
		self.rgb_path = os.path.join(self.object_dir, rgb_directory)
		os.mkdir(self.rgb_path)
		self.rgb_sub = rospy.Subscriber('/rgb/image_raw/compressed', CompressedImage, self.rgbCallback)

		## Uncomment below line to start depth data collection
		## Depth
		# self.depth_count = 0
		# depth_directory = 'depth'
		# self.depth_path = os.path.join(self.object_dir, depth_directory)
		# os.mkdir(self.depth_path)
		# self.depth_sub = rospy.Subscriber('/depth/image_raw', Image, self.depthCallback)

	def rgbCallback(self, rgb):
		try:
			# self.rgb = self.bridge.imgmsg_to_cv2(rgb, 'bgr8')
			np_arr = np.fromstring(rgb.data, np.uint8)
			self.rgb = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
			timestamp = utils.get_current_time_stamp()
			filename = 'rgb_'+str(self.rgb_count)+'_'+str(timestamp)+'.jpg'
			cv2.imwrite(self.rgb_path + '/' + filename, self.rgb)
			self.rgb_count += 1 
		except CvBridgeError, e:
			print(e)

	def depthCallback(self, depth):
		try:
			self.depth = self.bridge.imgmsg_to_cv2(depth, '32FC1')
			timestamp = utils.get_current_time_stamp()
			filename = 'depth_'+str(self.depth_count)+'_'+str(timestamp)+'.tif'
			imsave(self.depth_path + '/' + filename, self.depth)
			self.depth_count += 1
		except CvBridgeError, e:
			print(e)

	def stopRecord(self):
		self.rgb_sub.unregister()

		# Uncomment below line to stop depth data collection
		# self.depth_sub.unregister()

	def __str__(self):
		return 'Azure'


'''
Sensor: GelSight Tactile Sensor
Data: Tactile Images
Format: .jpg
'''
class GelSight(Sensor):
	def __init__(self, object_dir):
		self.object_dir = object_dir
		self.bridge = CvBridge()

		self.gelsight_count = 0
		gel_directory = 'gelsight'
		self.gel_path = os.path.join(self.object_dir, gel_directory)
		os.mkdir(self.gel_path)

		# First check the usb channel for the input video using 'ls -lrth /dev/video*'
		# Make the according changes in the usb_cam-test.launch for group: camera1 -> video_device/value ="/dev/video1"
		# Launch the file: roslaunch usb_cam usb_cam-test.launch
		self.gel_sub = rospy.Subscriber('/gelsight/usb_cam/image_raw', Image, self.gelSightCallback)

	def gelSightCallback(self, img):
		try:
			self.img = self.bridge.imgmsg_to_cv2(img, 'bgr8')
			timestamp = utils.get_current_time_stamp()
			filename = 'gelsight_'+str(self.gelsight_count)+'_'+str(timestamp)+'.jpg'
			cv2.imwrite(self.gel_path + '/' + filename, self.img)
			self.gelsight_count += 1
		except CvBridgeError, e:
			print(e)

	def stopRecord(self):
		self.gel_sub.unregister()

	def __str__(self):
		return 'GelSight'


'''
Sensor: WSG 50 Gripper
Data: WSG 50 Joint States (Position)
Format: .csv
'''
class WSG50(Sensor):
	def __init__(self, object_dir):
		self.object_dir = object_dir
		filename = 'gripper'
		self.wsg50_path = os.path.join(self.object_dir, '%s.csv'%(filename))

		self.position_elements = 1
		self.data_array = [0] * self.position_elements * 2
		self.wsg50_count = 0

		with open(self.wsg50_path, 'w') as csvfile:
			self.filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
			hdr = ['time', 'Left', 'Right']
			self.filewriter.writerow(hdr)

		self.wsg50_sub = rospy.Subscriber('/wsg_50_driver/joint_states', JointState, self.wsg50Callback)

	def wsg50Callback(self, wsg50_data):
		self.data_array[0] = wsg50_data.position[0]
		self.data_array[1] = wsg50_data.position[1]

		self._dumpWsg50DataToCSV()

	def _dumpWsg50DataToCSV(self):
		timestamp = utils.get_current_time_stamp()
		curr_datetime = timestamp
		with open(self.wsg50_path, 'a+') as csvfile:
			self.filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
			self.filewriter.writerow([curr_datetime] + self.data_array)
			self.wsg50_count += 1

	def stopRecord(self):
		self.wsg50_sub.unregister()

	def __str__(self):
		return 'WSG50'


'''
Sensor: UR5e
Data: Joint States and Joint Velocity
Format: .csv
'''
class Ur5e(Sensor):
	def __init__(self, object_dir):
		self.object_dir = object_dir
		filename = 'robot'
		self.ur5e_path = os.path.join(self.object_dir, '%s.csv'%(filename))

		self.joint_elements = 6
		self.data_array = [0] * self.joint_elements * 2
		self.ur5e_count = 0

		with open(self.ur5e_path, 'w') as csvfile:
			self.filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
			hdr = ['Time', 'Elbow', 'Shoulder Lift', 'Shoulder Pan', 'Wrist1', 'Wrist2', 'Wrist3', 'Velocity1', 'Velocity2', 'Velocity3', 'Velocity4', 'Velocity5', 'Velocity6']
			self.filewriter.writerow(hdr)

		self.ur5e_sub = rospy.Subscriber('/joint_states', JointState, self.ur5eCallback)

	def ur5eCallback(self, ur5e_data):
		for i in range(self.joint_elements):
			self.data_array[i] = ur5e_data.position[i]
		for i in range(self.joint_elements):
			self.data_array[i+6] = ur5e_data.velocity[i]

			self._dumpUr5eDataToCSV()

	def _dumpUr5eDataToCSV(self):
		timestamp = utils.get_current_time_stamp()
		curr_datetime = timestamp
		with open(self.ur5e_path, 'a+') as csvfile:
			self.filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
			self.filewriter.writerow([curr_datetime] + self.data_array)
			self.ur5e_count += 1

	def stopRecord(self):
		self.ur5e_sub.unregister()

	def __str__(self):
		return 'UR5e'


'''
Sensor: Logitech USB Camera
Data: Side Angle view
Format: .jpg
'''
class USBCAM(Sensor):
	def __init__(self, object_dir):
		self.object_dir = object_dir
		self.bridge = CvBridge()

		self.img_count1 = 0
		usb_directory = 'side_cam'
		self.usb_path1 = os.path.join(self.object_dir, usb_directory)
		os.mkdir(self.usb_path1)

		# First check the usb channel for the input video using 'ls -lrth /dev/video*'
		# Make the according changes in the usb_cam-test.launch for group: camera1 -> video_device/value ="/dev/video1"
		# Launch the file: roslaunch usb_cam usb_cam-test.launch
		self.usb_cam_sub1 = rospy.Subscriber('/camera1/usb_cam/image_raw', Image, self.usbCamCallback1)

	def usbCamCallback1(self, img):
		try:
			self.img1 = self.bridge.imgmsg_to_cv2(img, 'bgr8')
			timestamp = utils.get_current_time_stamp()
			filename = 'side_cam_'+str(self.img_count1)+'_'+str(timestamp)+'.jpg'
			cv2.imwrite(self.usb_path1 + '/' + filename, self.img1)
			self.img_count1 += 1
		except CvBridgeError, e:
			print(e)

	def stopRecord(self):
		self.usb_cam_sub1.unregister()

	def __str__(self):
		return 'USB_CAM'


'''
Sensor: OnRobot Force Torque Sensor 
Data: Force-Torque 6D Data Vector
Format: .csv
'''
class OnRobot(Sensor):
	def __init__(self, object_dir):
		self.object_dir = object_dir
		filename = 'f_t'		
		self.onrobot_path = os.path.join(self.object_dir, '%s.csv'%(filename))

		self.readings = 3
		self.data_array = [0] * self.readings * 2
		self.onrobot_count = 0

		with open(self.onrobot_path, 'w') as csvfile:
			self.filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
			hdr = ['time', 'Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz']
			self.filewriter.writerow(hdr)

		# First run the .cpp script to publish the topic
		# cd /home/okemo/cpp_ws
		# Run 'rosrun beginners_tutorials force_torque_usb'
		self.onrobot_sub = rospy.Subscriber('/ur5e/onrobot_force_torque_usb_1', Wrench, self.ftCallback)

	def ftCallback(self, ft_data):
		self.data_array[0] = ft_data.force.x
		self.data_array[1] = ft_data.force.y
		self.data_array[2] = ft_data.force.z

		self.data_array[3] = ft_data.torque.x
		self.data_array[4] = ft_data.torque.y
		self.data_array[5] = ft_data.torque.z

		self._dumpOnRobotDataToCSV()

	def _dumpOnRobotDataToCSV(self):
		timestamp = utils.get_current_time_stamp()
		curr_datetime = timestamp

		with open(self.onrobot_path, 'a+') as csvfile:
			self.filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
			self.filewriter.writerow([curr_datetime] + self.data_array)
			self.onrobot_count += 1

	def stopRecord(self):
		self.onrobot_sub.unregister()

	def __str__(self):
		return 'OnRobot'

'''
Sensor: Azure Kinect [Overhead] [Using as a normal USB cam]
Data: RGB Raw
Format: .jpg
'''
class UsbCam2(Sensor):
	def __init__(self, object_dir):
		self.object_dir = object_dir
		self.bridge = CvBridge()

		self.img_count2 = 0
		usb_directory = 'top_cam'
		self.usb_path2 = os.path.join(self.object_dir, usb_directory)
		os.mkdir(self.usb_path2)

		# First check the usb channel for the input video using 'ls -lrth /dev/video*'
		# Make the according changes in the usb_cam-test.launch for group: camera1 -> video_device/value ="/dev/video1"
		# Launch the file: roslaunch usb_cam usb_cam-test.launch
		self.usb_cam_sub2 = rospy.Subscriber('/camera2/usb_cam/image_raw', Image, self.usbCamCallback2)

	def usbCamCallback2(self, img):
		try:
			self.img2 = self.bridge.imgmsg_to_cv2(img, 'bgr8')
			timestamp = utils.get_current_time_stamp()
			filename = 'top_cam_'+str(self.img_count2)+'_'+str(timestamp)+'.jpg'
			cv2.imwrite(self.usb_path2 + '/' + filename, self.img2)
			self.img_count2 += 1
		except CvBridgeError, e:
			print(e)

	def stopRecord(self):
		self.usb_cam_sub2.unregister()

	def __str__(self):
		return 'Kinect USB Cam'


# Helper functions
def start_data_collection(trial_dir):
	azure = Azure(trial_dir)
	tactile = GelSight(trial_dir)
	wsg50 = WSG50(trial_dir)
	usbcam = USBCAM(trial_dir)
	onrobot = OnRobot(trial_dir)
	ur5e = Ur5e(trial_dir)

	return azure, tactile, wsg50, usbcam, onrobot, ur5e

def stop_data_collection(azure, tactile, wsg50, usbcam, onrobot, ur5e):
	azure.stopRecord()
	tactile.stopRecord()
	wsg50.stopRecord()
	usbcam.stopRecord()	
	onrobot.stopRecord()
	ur5e.stopRecord()