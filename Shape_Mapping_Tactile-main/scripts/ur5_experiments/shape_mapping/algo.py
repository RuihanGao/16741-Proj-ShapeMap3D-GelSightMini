#!/usr/bin/env python 

'''
Shubham Kanitkar (skanitka@andrew.cmu.edu) June, 2021
'''
from record import collect_frame, start_data_collection, stop_data_collection
import numpy as np 
from numpy import linalg as LA
import math 
import pdb 
from scipy.spatial.transform import Rotation as R
import time 

NUM_HEIGHTS = 5	# number of heights  
NUM_ANGLES = 8	# number of angles 
NUM_TOP = 5		# number of touches on top 
NUM_INTERP = 50 # number of interpolation
SAFE_RADIUS = 20.0/100.0 # 15cm safe radius 
HOMING_X = 0.131
HOMING_Y = 0.471
SAFE_OFFSET = 6.0/100.0   # 5cm offset from base (gelsight doesnt hit clamp)
GRIPPER2GELPAD_Z = 7.2/100.0 # Z offset to the gelpad (between gripper bottom and gelpad center)

# Touch object at set heights and angles 
def explore_object(robot, gripper, object, trial_dir):

	# get object properties
	object_name = object["name"]
	object_height = object["height"]/1000.0
	object_pose = np.array(object["pose"])

	SAFE_HEIGHT = object_height + GRIPPER2GELPAD_Z   + 10.0/100.0 # 15cm safe radius 

	azure, tactile, usbcam, ur5e = start_data_collection(trial_dir)	# Start data collection
	print('\nStep 1: capture one depth + RGB image and save\n')
	collect_frame(azure=azure) 

	print('\nStep 2: Compute the trajectory\n')
	# symmetric objects 
	if (object_name  == "021_bleach_cleanser"):
		thetas = np.radians( np.array([0.0, 60.0, 90.0, 120.0, 180.0, 240.0, 270.0, 300.0]) )
	else:
		thetas = np.linspace(0.0, 2*np.pi, num=NUM_ANGLES, endpoint=False)# approach angles (wrt object and robot)

	xs = SAFE_RADIUS * np.cos(thetas) + object_pose[0]					# approach x (wrt robot)
	ys = -SAFE_RADIUS * np.sin(thetas) + object_pose[1]					# approach y (wrt robot)
	thetas +=  np.pi	# correct for end effector -> gelsight transform	
	thetas = (thetas + np.pi) % (2 * np.pi) - np.pi						# wrap angles 
	zs = np.linspace(object_height, SAFE_OFFSET, num=NUM_HEIGHTS, endpoint=False)	# approach z (wrt robot)
	zs = zs + GRIPPER2GELPAD_Z
	zs = zs + object_pose[2]

	print('Exploration parameters:')
	print('NUM_HEIGHTS: ' + str(NUM_HEIGHTS))
	print('NUM_TOP: ' + str(NUM_TOP))
	print('NUM_INTERP: ' + str(NUM_INTERP))
	print('SAFE_RADIUS: ' + str(SAFE_RADIUS))
	print('SAFE_OFFSET: ' + str(SAFE_OFFSET))
	print('object_height: ' + str(object_height))
	print('object_position: [' + str(object_pose[0]) + ', ' + str(object_pose[1])  + ', ' + str(object_pose[2])  + ']')

	direction_count = 0
	height_count = 0
	print('Step 3: Exploration cycle')
	for x, y, theta in zip(xs, ys, thetas):
		direction_count += 1
		print('\nStep 3.1: Direction sweep #' + str(direction_count) + '\n')

		print('\nStep 3.1.1: Joint-angle homing\n')
		robot.joint_angle_homing()													


		print('\nStep 3.1.2: Rotating wrist\n')
		robot.print_current_rotation()
		robot.rotate_wrist(theta)

		print('Moving to safe height\n')
		robot.move_to_location(HOMING_X, HOMING_Y, SAFE_HEIGHT)
		robot.move_to_location(object_pose[0], object_pose[1], SAFE_HEIGHT)
		robot.move_to_location(x, y, SAFE_HEIGHT)

		# loop over multiple heights
		for z in zs:

			height_count += 1
			target_safe_position = np.array([x, y, z])
			target_goal_position = np.array([object_pose[0], object_pose[1], z])

			small_step = 0.001*(target_goal_position - target_safe_position)/LA.norm(target_goal_position - target_safe_position)
			large_step = (SAFE_RADIUS/NUM_INTERP)*(target_goal_position - target_safe_position)/LA.norm(target_goal_position - target_safe_position)

			interp_position = np.stack([np.linspace(i,j,NUM_INTERP) for i,j in zip(target_safe_position,target_goal_position)],axis=1)			# interpolate 
			
			# pdb.set_trace()
			print('\nStep 3.1.3.A: Height sweep #' + str(height_count) + ' Location: ' + 
					'[' + str(interp_position[0,0]) + ', ' + str(interp_position[0,1])  + ', ' + str(interp_position[0,2])  + ']\n')
			robot.move_to_location(interp_position[0, 0], interp_position[0, 1], interp_position[0, 2])

			time.sleep(2.0)
			tactile.saveBackground()
			collect_frame(tactile=tactile, wsg50=None, ur5e=ur5e) 			# collect one sample upon contact 

			interp_count = 1
			no_contact = False
			print('\nStep 3.1.3.B: Moving towards object (override y/n)...\n')
			# contact detection algorithm with gripper.get_status()
			while (not tactile.detectContact()):
				print(interp_count)
				if (interp_count == (interp_position.shape[0])):
					print('\nStep 3.1.3.C: Exceeded maximum distance, no contact!\n')
					no_contact = True
					break
				time.sleep(0.5)
				robot.move_offset_distance(offset_x=large_step[0], offset_y=large_step[1], offset_z=large_step[2], override=True)
				interp_count += 1

			if no_contact == False:
				print('\nStep 3.1.3.D: Made contact, moving 1mm!\n')
				robot.move_offset_distance(offset_x=small_step[0], offset_y=small_step[1], offset_z=small_step[2], override=True)
				collect_frame(tactile=tactile, wsg50=None, ur5e=ur5e) 			# collect one sample upon contact 

			print('\nStep 3.1.3.E: Retracing path...\n')
			robot.move_to_location(interp_position[0, 0], interp_position[0, 1], interp_position[0, 2], override=True)
		
		print('Moving to safe height\n')
		robot.move_to_location(x, y, SAFE_HEIGHT)
		robot.move_to_location(object_pose[0], object_pose[1], SAFE_HEIGHT)
		robot.move_to_location(HOMING_X, HOMING_Y, SAFE_HEIGHT)

	# TODO
	# print('\nStep 3.4: Top exploration\n')
	# horizontal homing and rotate wrist 
	# do downwards motion until contact 

	stop_data_collection(azure=azure, tactile=tactile, wsg50=None, usbcam=usbcam, ur5e=ur5e)				# Stop data collection

	return

# Touch object at set heights and angles 
def explore_object_rect(robot, gripper, object, trial_dir):

	# get object properties
	object_name = object["name"]
	object_height = object["height"]/1000.0
	object_pose = np.array(object["pose"])

	SAFE_HEIGHT = object_height + GRIPPER2GELPAD_Z   + 10.0/100.0 # 15cm safe radius 

	azure, tactile, usbcam, ur5e = start_data_collection(trial_dir)	# Start data collection
	print('\nStep 1: capture one depth + RGB image and save\n')
	collect_frame(azure=azure) 

	if object_name == "004_sugar_box":
		a = 38.5/1000.0 # x dimension
		b = 90.0/1000.0 # y dimension 
		xs = np.array([SAFE_RADIUS, SAFE_RADIUS, a/4, -a/4, -SAFE_RADIUS, -SAFE_RADIUS, -a/4, a/4]) + object_pose[0]					# approach x (wrt robot)
		ys = np.array([b/4, -b/4, -SAFE_RADIUS, -SAFE_RADIUS, -b/4, b/4, SAFE_RADIUS, SAFE_RADIUS]) + object_pose[1]					# approach x (wrt robot)
	elif object_name == "010_potted_meat_can":
		a = 51.5/1000.0 # x dimension
		b = 97.0/1000.0 # y dimension 
		xs = np.array([SAFE_RADIUS, SAFE_RADIUS, a/4, -a/4, -SAFE_RADIUS, -SAFE_RADIUS, -a/4, a/4]) + object_pose[0]					# approach x (wrt robot)
		ys = np.array([b/4, -b/4, -SAFE_RADIUS, -SAFE_RADIUS, -b/4, b/4, SAFE_RADIUS, SAFE_RADIUS]) + object_pose[1]					# approach x (wrt robot)
	else:
		a = 91.0/1000.0 # x dimension
		b = 91.0/1000.0 # y dimension 
		xs = np.array([SAFE_RADIUS, SAFE_RADIUS, a/4, -a/4, -SAFE_RADIUS, -SAFE_RADIUS, -a/4, a/4]) + object_pose[0]					# approach x (wrt robot)
		ys = np.array([b/4, -b/4, -SAFE_RADIUS, -SAFE_RADIUS, -b/4, b/4, SAFE_RADIUS, SAFE_RADIUS]) + object_pose[1]					# approach x (wrt robot)

	print('\nStep 2: Compute the trajectory\n')
	thetas = np.radians( np.array([0.0, 0.0, 90.0, 90.0, 180.0, 180.0, 270.0, 270.0]) )
	thetas +=  np.pi	# correct for end effector -> gelsight transform	
	thetas = (thetas + np.pi) % (2 * np.pi) - np.pi						# wrap angles 
	zs = np.linspace(object_height, SAFE_OFFSET, num=NUM_HEIGHTS, endpoint=False)	# approach z (wrt robot)
	zs = zs + GRIPPER2GELPAD_Z
	zs = zs + object_pose[2]

	print('RECT Exploration parameters:')
	print('NUM_HEIGHTS: ' + str(NUM_HEIGHTS))
	print('NUM_TOP: ' + str(NUM_TOP))
	print('NUM_INTERP: ' + str(NUM_INTERP))
	print('SAFE_RADIUS: ' + str(SAFE_RADIUS))
	print('SAFE_OFFSET: ' + str(SAFE_OFFSET))
	print('object_height: ' + str(object_height))
	print('object_position: [' + str(object_pose[0]) + ', ' + str(object_pose[1])  + ', ' + str(object_pose[2])  + ']')

	direction_count = 0
	height_count = 0
	print('Step 3: Exploration cycle')
	for x, y, theta in zip(xs, ys, thetas):
		direction_count += 1
		print('\nStep 3.1: Direction sweep #' + str(direction_count) + '\n')

		print('\nStep 3.1.1: Joint-angle homing\n')
		robot.joint_angle_homing()													


		print('\nStep 3.1.2: Rotating wrist\n')
		robot.print_current_rotation()
		robot.rotate_wrist(theta)

		print('Moving to safe height\n')
		robot.move_to_location(HOMING_X, HOMING_Y, SAFE_HEIGHT)
		robot.move_to_location(object_pose[0], object_pose[1], SAFE_HEIGHT)
		robot.move_to_location(x, y, SAFE_HEIGHT)

		# loop over multiple heights
		for z in zs:

			height_count += 1
			target_safe_position = np.array([x, y, z])

			if ( (direction_count == 1) or (direction_count == 2) or (direction_count == 5) or (direction_count == 6) ):
				target_goal_position = np.array([object_pose[0], y, z])
			else: 
				target_goal_position = np.array([x, object_pose[1], z])

			small_step = 0.001*(target_goal_position - target_safe_position)/LA.norm(target_goal_position - target_safe_position)
			large_step = (SAFE_RADIUS/NUM_INTERP)*(target_goal_position - target_safe_position)/LA.norm(target_goal_position - target_safe_position)

			interp_position = np.stack([np.linspace(i,j,NUM_INTERP) for i,j in zip(target_safe_position,target_goal_position)],axis=1)			# interpolate 
			
			# pdb.set_trace()
			print('\nStep 3.1.3.A: Height sweep #' + str(height_count) + ' Location: ' + 
					'[' + str(interp_position[0,0]) + ', ' + str(interp_position[0,1])  + ', ' + str(interp_position[0,2])  + ']\n')
			robot.move_to_location(interp_position[0, 0], interp_position[0, 1], interp_position[0, 2])

			time.sleep(2.0)
			tactile.saveBackground()
			collect_frame(tactile=tactile, wsg50=None, ur5e=ur5e) 			# collect one sample upon contact 

			interp_count = 1
			no_contact = False
			print('\nStep 3.1.3.B: Moving towards object (override y/n)...\n')
			# contact detection algorithm with gripper.get_status()
			while (not tactile.detectContact()):
				print(interp_count)
				if (interp_count == (interp_position.shape[0])):
					print('\nStep 3.1.3.C: Exceeded maximum distance, no contact!\n')
					no_contact = True
					break
				time.sleep(0.5)
				robot.move_offset_distance(offset_x=large_step[0], offset_y=large_step[1], offset_z=large_step[2], override=True)
				interp_count += 1

			if no_contact == False:
				print('\nStep 3.1.3.D: Made contact, moving 1mm!\n')
				robot.move_offset_distance(offset_x=small_step[0], offset_y=small_step[1], offset_z=small_step[2], override=True)
				collect_frame(tactile=tactile, wsg50=None, ur5e=ur5e) 			# collect one sample upon contact 

			print('\nStep 3.1.3.E: Retracing path...\n')
			robot.move_to_location(interp_position[0, 0], interp_position[0, 1], interp_position[0, 2], override=True)
		
		print('Moving to safe height\n')
		robot.move_to_location(x, y, SAFE_HEIGHT)
		robot.move_to_location(object_pose[0], object_pose[1], SAFE_HEIGHT)
		robot.move_to_location(HOMING_X, HOMING_Y, SAFE_HEIGHT)

	# TODO
	# print('\nStep 3.4: Top exploration\n')
	# horizontal homing and rotate wrist 
	# do downwards motion until contact 

	stop_data_collection(azure=azure, tactile=tactile, wsg50=None, usbcam=usbcam, ur5e=ur5e)				# Stop data collection

	return