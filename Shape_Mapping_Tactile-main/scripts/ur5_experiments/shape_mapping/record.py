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
import tf

from skimage.io import imsave
from cv_bridge import CvBridge, CvBridgeError
from datetime import datetime

from sensor_msgs.msg import Image, JointState, CompressedImage, PointCloud2
from tf2_msgs.msg import TFMessage
import sensor_msgs.point_cloud2 as pc2
import json 

from geometry_msgs.msg import Wrench
import pdb 

class Sensor:
	def __init__(self, object_dir):
		self.object_dir = object_dir

	def __str__(self):
		return 'General'

class Transforms: 
	def __init__(self, object_dir):
		object_dir = object_dir
		world2azure =  np.array([-0.754, 0.618, 0.790, 0.645, -0.636, 0.298, -0.302]) # from calib.launch file
		world2object = np.array([-0.10, 0.55, 0.039, 0.0, 0.0, 0.0, 1.0]) # from objects.py
		gripper2gelsight = np.array([7.2/100.0, -(0.05449 + 0.055), 0.0, 0.7071068, 0, 0, 0.7071068]) # measured by hand and from gripper feedback

		data = { 'world2azure' : world2azure.tolist(), 
			   'world2object' : world2object.tolist(), 
			   'gripper2gelsight' : gripper2gelsight.tolist() }

		filename = os.path.join(object_dir, "tf.json")
		with open(filename, 'w') as f:
			json.dump(data, f)

		print('Transforms saved at:' + filename)

'''
Sensor: Azure Kinect [Side View]
Data: RGB Compressed and RGB-D Raw
Format: .jpg
'''
class Azure(Sensor):
	def __init__(self, object_dir):
		self.object_dir = object_dir
		self.bridge = CvBridge()

		self.saveDepthFlag = False
		self.savePointFlag = False

		# RGB
		self.rgb_count = 0
		rgb_directory = 'rgb'
		self.rgb_path = os.path.join(self.object_dir, rgb_directory)
		os.mkdir(self.rgb_path)
		self.rgb_sub = rospy.Subscriber('/rgb/image_raw/compressed', CompressedImage, self.rgbCallback)

		## Depth
		self.depth_count = 0
		depth_directory = 'depth'
		self.depth_path = os.path.join(self.object_dir, depth_directory)
		os.mkdir(self.depth_path)
		self.depth_sub = rospy.Subscriber('/depth/image_raw', Image, self.depthCallback)

		## PointCloud2
		self.point_count = 0
		point_directory = 'pc'
		self.point_path = os.path.join(self.object_dir, point_directory)
		os.mkdir(self.point_path)
		self.point_sub = rospy.Subscriber('/points2', PointCloud2, self.pointCallback)

	def rgbCallback(self, rgb):
		try:
			# self.rgb = self.bridge.imgmsg_to_cv2(rgb, 'bgr8')
			np_arr = np.fromstring(rgb.data, np.uint8)
			self.rgb = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
			timestamp = utils.get_current_time_stamp()
			filename = 'rgb_'+str(self.rgb_count)+'_'+str(timestamp)+'.jpg'
			cv2.imwrite(self.rgb_path + '/' + filename, self.rgb)
			if (self.rgb_count % 1000 == 0):
				print('Azure RGB #' + str(self.rgb_count) + ' saved at: ' + self.rgb_path + '/' + filename)
			self.rgb_count += 1 

		except CvBridgeError, e:
			print(e)
		
	def depthCallback(self, depth):
		try:
			self.depth = self.bridge.imgmsg_to_cv2(depth, '32FC1')
			timestamp = utils.get_current_time_stamp()
			if self.saveDepthFlag:
				filename = 'depth_'+str(self.depth_count)+'_'+str(timestamp)+'.tif'
				imsave(self.depth_path + '/' + filename, self.depth)
				print('Azure Depth #' + str(self.depth_count) + ' saved at: ' + self.depth_path + '/' + filename)
				self.depth_count += 1
				self.saveDepthFlag = False

		except CvBridgeError, e:
			print(e)

	def pointCallback(self, cloud):
		'''https://answers.ros.org/question/344096/subscribe-pointcloud-and-convert-it-to-numpy-in-python/'''
		if self.savePointFlag:
			gen = pc2.read_points(cloud, skip_nans=True)
			int_data = list(gen)
			array_gen = np.array(int_data)[:,0:3]
			timestamp = utils.get_current_time_stamp()
			filename = 'pc_'+str(self.point_count)+'_'+str(timestamp)+'.npy'
			with open(self.point_path + '/' + filename, 'wb') as f:
				np.save(f, array_gen)
			print('Azure PC #' + str(self.point_count) + ' saved at: ' + self.point_path + '/' + filename)
			self.point_count += 1 
			self.savePointFlag = False

	def stopRecord(self):
		self.rgb_sub.unregister()
		self.depth_sub.unregister()
		self.point_sub.unregister()

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

		self.saveFlag = False
		self.background = None
		self.contactMask = None
		self.MaxImg = None
		self.DiffImg = None 

		self.IntensityThreshold = 2000 # 480*640 is max area
		self.pixelThreshold = 30

		self.gelsight_count = 0
		self.gelsight_save_count = 0

		gel_directory = 'gelsight'
		self.gel_path = os.path.join(self.object_dir, gel_directory)
		os.mkdir(self.gel_path)
		print("Gelsight data collection is created!")

		# First check the usb channel for the input video using 'ls -lrth /dev/video*'
		# Make the according changes in the usb_cam-test.launch for group: camera1 -> video_device/value ="/dev/video1"
		# Launch the file: roslaunch usb_cam usb_cam-test.launch
		self.gel_sub = rospy.Subscriber('/gelsight/usb_cam/image_raw', Image, self.gelSightCallback)

	def gelSightCallback(self, img):
		try:
			self.img = self.bridge.imgmsg_to_cv2(img, 'bgr8')
			timestamp = utils.get_current_time_stamp()

			if self.saveFlag:
				filename = 'gelsight_'+str(self.gelsight_save_count)+'_'+str(timestamp)+'.jpg'
				cv2.imwrite(self.gel_path + '/' + filename, self.img)
				print('Gelsight RGB #' + str(self.gelsight_save_count) + ' saved at: ' + self.gel_path + '/' + filename)
				# filename = 'contactmask_'+str(self.gelsight_save_count)+'_'+str(timestamp)+'.jpg'
				# cv2.imwrite(self.gel_path + '/' + filename, self.contactMask * 255)
				# print('Contact mask RGB #' + str(self.gelsight_save_count) + ' saved at: ' + self.gel_path + '/' + filename)
				self.gelsight_save_count += 1
				self.saveFlag = False

		except CvBridgeError, e:
			print(e)

	def debugContact(self):
		cv2.imshow("self.img", self.img)
		cv2.imshow("self.MaxImg", self.MaxImg)
		cv2.imshow("self.DiffImg", (self.DiffImg/256).astype('uint8'))
		cv2.imshow("self.contactMask", self.contactMask)
		cv2.waitKey(0)

	def saveBackground(self):
		while(True):
			try:
				self.background = np.int16(self.img.copy())
				break
			except: 
				pass
	def detectContact(self):
		# compare self.img with self.background 
		self.DiffImg = np.abs(np.int16(self.img) - self.background)
		self.MaxImg = np.max(self.DiffImg, 2)
		CountNum = 1*(self.MaxImg>self.pixelThreshold).sum()
		# print('Max of self.img: ', np.max(self.img))
		# print('Min of self.img: ', np.min(self.img))
		# print('Max of self.background: ', np.max(self.background))
		# print('Min of self.background: ', np.min(self.background))
		# print('CountNum: ', CountNum)

		if CountNum > self.IntensityThreshold :
			# cv2.imwrite(self.gel_path + '/' + 'diffimg.jpg', self.DiffImg)
			# cv2.imwrite(self.gel_path + '/' + 'maximg.jpg', MaxImg)
			print('\n \n $$$$$$$$$$$$$$$$$$$ CONTACT $$$$$$$$$$$$$$$$$$$ \n \n') 
			self.MaxImg[self.MaxImg<self.pixelThreshold] = 0

			sec95C = np.percentile(self.MaxImg, 95)
			self.contactMask = self.MaxImg > 0.9*sec95C

			# kernel = np.ones((5,5),np.uint8)
			kernel = np.array([[0,0,1,0,0],[0,1,1,1,0],[1,1,1,1,1],[0,1,1,1,0],[0,0,1,0,0]],dtype=np.uint8)
			self.contactMask = cv2.dilate(np.float32(self.contactMask),kernel,iterations = 1)

			self.MaxImg = cv2.normalize(self.MaxImg, None, 255,0, cv2.NORM_MINMAX, cv2.CV_8UC1)
			# pdb.set_trace()
			return True 
		else:
			return False
	
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

		self.saveFlag = False

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

		if self.saveFlag:
			self._dumpWsg50DataToCSV()
			self.saveFlag = False

	def _dumpWsg50DataToCSV(self):
		timestamp = utils.get_current_time_stamp()
		curr_datetime = timestamp
		with open(self.wsg50_path, 'a+') as csvfile:
			self.filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
			self.filewriter.writerow([curr_datetime] + self.data_array)
			print('Gripper positions #' + str(self.wsg50_count) + ' saved at: ' + self.wsg50_path)
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

		self.saveFlag = False

		self.pose_elements = 7
		self.data_array = [0] * self.pose_elements
		self.ur5e_count = 0

		with open(self.ur5e_path, 'w') as csvfile:
			self.filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
			hdr = ['Time', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']
			self.filewriter.writerow(hdr)
	
		self.listener = tf.TransformListener()
		self.ur5e_sub = rospy.Subscriber('/tf', TFMessage, self.ur5eCallback)

	def ur5eCallback(self, ur5e_data):
		if self.saveFlag:
			try:
				(trans,rot) = self.listener.lookupTransform('/world', '/gripper', rospy.Time(0))
				self.data_array[0] = trans[0]
				self.data_array[1] = trans[1]
				self.data_array[2] = trans[2]
				self.data_array[3] = rot[0]
				self.data_array[4] = rot[1]
				self.data_array[5] = rot[2]
				self.data_array[6] = rot[3]
				self.saveFlag = False
			except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
				print("ERROR: No world2gripper TF, saving 0 array")

			self._dumpUr5eDataToCSV()

	def _dumpUr5eDataToCSV(self):
		'''saved as x, y, z, qx, qy, qz, qw'''
		timestamp = utils.get_current_time_stamp()
		curr_datetime = timestamp
		with open(self.ur5e_path, 'a+') as csvfile:
			self.filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
			self.filewriter.writerow([curr_datetime] + self.data_array)
			print('Gripper pose #' + str(self.ur5e_count) + ' saved at: ' + self.ur5e_path)
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
			if (self.img_count1 % 1000 == 0):
				print('USBCAM RGB #' + str(self.img_count1) + ' saved at: ' + self.usb_path1 + '/' + filename)
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

		self.saveFlag = False

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
			if self.saveFlag:
				filename = 'top_cam_'+str(self.img_count2)+'_'+str(timestamp)+'.jpg'
				cv2.imwrite(self.usb_path2 + '/' + filename, self.img2)
				self.img_count2 += 1
				self.saveFlag = False

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
	# wsg50 = WSG50(trial_dir)
	usbcam = USBCAM(trial_dir)
	ur5e = Ur5e(trial_dir)
	Transforms(trial_dir)

	# return azure, tactile, wsg50, usbcam, ur5e
	return azure, tactile, usbcam, ur5e

def stop_data_collection(azure=None, tactile=None, wsg50=None, usbcam=None, ur5e=None):
	if azure:
		azure.stopRecord()
	if tactile:
		tactile.stopRecord()
	if wsg50:
		wsg50.stopRecord()
	if usbcam:
		usbcam.stopRecord()	
	if ur5e:
		ur5e.stopRecord()

def collect_frame(azure=None, tactile=None, wsg50=None, ur5e=None):
	if azure:
		azure.saveDepthFlag = True
		azure.savePointFlag = True
	if tactile:
		tactile.saveFlag = True
	if wsg50:
		wsg50.saveFlag = True
	if ur5e:
		ur5e.saveFlag = True
