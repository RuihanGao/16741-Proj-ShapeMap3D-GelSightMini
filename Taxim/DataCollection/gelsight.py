import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

'''
Sensor: GelSight Tactile Sensor
Data: Tactile Images
Format: .jpg
'''
class GelSight:
	def __init__(self, object_dir):
		self.object_dir = object_dir
		self.bridge = CvBridge()

		self.gelsight_count = 0
		self.gel_path = object_dir


		# First check the usb channel for the input video using 'ls -lrth /dev/video*'
		# Make the according changes in the usb_cam-test.launch for group: camera1 -> video_device/value ="/dev/video1"
		# Launch the file: roslaunch usb_cam usb_cam-test.launch

		print("reading in the gelsight images...")
		# self.gel_sub = rospy.Subscriber('/usb_cam/image_raw', Image, self.gelSightCallback)
		# RH: since we change to GelSight mini, instead of using usb_cam package, we subscribe to the rostopic published by gs mini
		self.gel_sub = rospy.Subscriber('/gsmini_rawimg_0', Image, self.gelSightCallback)

	def gelSightCallback(self, img):
		try:
			self.img = self.bridge.imgmsg_to_cv2(img, desired_encoding='passthrough')
		except CvBridgeError as e:
			print(e)

	def stopRecord(self):
		self.gel_sub.unregister()

	def __str__(self):
		return 'GelSight'
