#!/usr/bin/env python 

'''
Shubham Kanitkar (skanitka@andrew.cmu.edu) April, 2021
'''

import os
import time
import csv
from datetime import datetime
import numpy as np
from scipy.spatial.transform import Rotation as R

# translate then rotate
T_EE2GELPAD = [19.0, 10.75, 0.0]
T_GELPAD2EE = [-19.0, -10.75, -0.0]
R_EE2GELPAD = R.from_euler('x', 90, degrees=True)  # [0.5253, 0.8509, 0.0, 0.0]
R_GELPAD2EE = R_EE2GELPAD.inv()

def select_operation():
	while True:
		try:
			print('Select Operation [1/2]:')
			print('1. Robot Movement')
			print('2. Data Collection')
			operation = raw_input()
		except ValueError:
			print('Invalid selection. Please try again')
			continue
		if operation not in ['1', '2']:
			print('Invalid selection! Select from the given options')
			continue
		else:
			break
	return operation

def get_current_time_stamp():
	now = datetime.now()
	x = np.datetime64(now)
	timestamp = x.view('i8')
	return int(timestamp)	

# get user input name and load height, pose from objects.py
def load_object_properties(hardcode = False):
	from objects import objectDict
	object = dict()
	while len(object) == 0:
		if hardcode:
			objectName = "002_master_chef_can"
		else:
			objectName = raw_input('Enter name of object:')	
		print(objectName)				    
		if objectName in objectDict:
			object["name"] = objectName
			object["height"] = objectDict[objectName][0]	                                
			object["pose"] = objectDict[objectName][1]	                                       
		else:
			print('Invalid name, try again...')
			pass
	return object 
		
# enter object name, and create directory inside parent_dir
def create_data_dir(parent_dir, object_name):
	timestamp = get_current_time_stamp()
	trial_path = object_name + '_' + str(timestamp)
	trial_dir = os.path.join(parent_dir, trial_path)
	os.mkdir(trial_dir)
	print('Current Object Directory:')
	print(trial_dir)

	return trial_dir, trial_path

# User Input Handlers
def robot_movement_direction_user_input():
	while True:
		try:
			print('Select Moverment Direction: [UP: \'w\', DOWN: \'s\', LEFT: \'a\', RIGHT: \'d\', FRONT: \'z\', BACK: \'x\', GRASP: \'g\', RELEASE: \'r\']')
			direction = raw_input()
		except ValueError:
			print('Invalid selection! Please try again')
			continue
		if direction not in ['w', 's', 'a', 'd', 'z', 'x', 'g', 'r']:
			print('Invalid selection! Select from the given options')
			continue
		else:
			break
	return direction  

def reached_destination():
	while True:
		try:
			print('Reached Destination? [y/n]')
			out = raw_input()
		except ValueError:
			print('Invalid selection. Please try again')
			continue
		if out not in ['y', 'n']:
			print('Invalid selection! Select from the given options')
			continue
		else:
			break
	if out == 'y':
		return True
	return False


# Data Annotation Helper Functions
def _label(swipe, writer):
	# choices = []
	while True:
		choice = int(raw_input('Label: Enter your choice [0/1/2/3]- '))
		if choice in [0, 1, 2, 3]:
			# choices.append(choice)
			if choice == 0:
				writer.writerow([swipe, 'blank'])
			elif choice == 1:
				writer.writerow([swipe, 'miss'])
			elif choice == 2:
				writer.writerow([swipe, 'insert'])
			elif choice == 3:
				writer.writerow([swipe, 'not present'])
			break
		else:
			print('Invalid Selection')
			continue
	return choice

def label_data(trial_dir, v):
	label_path = os.path.join(trial_dir, 'label.csv')

	print('*'*30)
	print('Data Labelling: ')
	print('0. Blank')
	print('1. Miss')
	print('2. Insert')
	print('3. Not Present')

	f = open(label_path, 'a+')
	writer = csv.writer(f)
	
	swipe = 'swipe_' + str(v)
	choice = _label(swipe, writer)
	# writer.writerow(['force', force])from_euler
# CHECK if this works 
# Converts transforms between robot - gelpad, to robot - EE
def gelpad2ee(gelpad_t, gelpad_r):
	ee_t = gelpad_t + T_GELPAD2EE
	ee_r = R_GELPAD2EE * R.from_quat([gelpad_r[1], gelpad_r[2], gelpad_r[3], gelpad_r[0]])
	return ee_t, ee_r.as_quat()

# Converts transforms between robot - EE, to robot - gelpad
def ee2gelpad(ee_t, ee_r):
	gelpad_t = ee_t + T_GELPAD2EE
	gelpad_r = R_EE2GELPAD * R.from_quat([ee_r[1], ee_r[2], ee_r[3], ee_r[0]])
	return gelpad_t, gelpad_r.as_quat()