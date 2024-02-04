import matplotlib.pyplot as plt
import numpy as np
import os
from os import path as osp
import cv2
import scipy

import sys
sys.path.append("..")
import basics.sensorParams as psp

# obj = 'bunny'
obj = 'mustard_bottle'
# obj = 'power_drill'
# data_path = osp.join('..','..','..','data',obj,obj+'_50sampled.mat')

save_path = osp.join('..', '..','..','..','processed_data',obj)
save_folder = osp.join(save_path,'gt_height_map')
pose_file = open(osp.join(save_path,'pose.txt'),'r')

all_heightMaps = pose_file.readlines()
print("Total # "+ str(len(all_heightMaps)) + " height maps.")
print("#...orientation(quaternion)...position in object's original coords and in original scale")
for i in range(len(all_heightMaps)):
    print("# iter: " + str(i))
    line = all_heightMaps[i].split(',')
    img_name = line[0]
    qr = line[1]
    t = line[2]
    print(qr, t)
    img = np.load(osp.join(save_folder,img_name))
    plt.imshow(img)
    plt.show()

pose_file.close()
