#!/usr/bin/env python 

'''
Shubham Kanitkar (skanitka@andrew.cmu.edu) June, 2021
'''


# Movement cycle function - Algorithm implementaion
def movement_cycle(trial_dir, log_dir, robot, gripper):

	# Grasp and lift the object
	gripper.graspWithForce()
	robot.move_offset_distance(offset_x=0, offset_y=0, offset_z=0.06)

	# Manipulation
	while True:
		print('Select Manipulation Method: ')
		print('1. Manual Control')
		print('2. Automatic')

		method = int(raw_input())
		if method in [1, 2]:
			if method == 1:
				print('Robot working in the Manual Mode')
				robot.manual_control(gripper)
				break
			elif method == 2:
				print('Robot working in the Automatic Mode')
				print('Implement your logic here')

				# Implement your logic here
				# Take a look at helper functions in robot.py 
				break

		else:
			print('Invalid Selection')
			continue

	# Release the object
	gripper.homing()