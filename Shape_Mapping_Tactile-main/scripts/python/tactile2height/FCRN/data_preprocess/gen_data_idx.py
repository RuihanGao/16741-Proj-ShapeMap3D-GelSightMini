import numpy as np
import os
from os import path as osp
np.random.seed(1)

# change the data_root to the root directory of dataset with all objects

# data_root_path = osp.join('..','..','..','dataset')
data_root_path = osp.join('/media','datadisk','zsi','shape_mapping','data')
object_folders = sorted(os.listdir(data_root_path))

# write training/validation/testing data loader files
save_path = osp.join('/home','zsi','shape_mapping','dataset')
train_data_file = open(osp.join(save_path,'train_data.txt'),'w')
train_label_file = open(osp.join(save_path,'train_label.txt'),'w')

dev_data_file = open(osp.join(save_path,'dev_data.txt'),'w')
dev_label_file = open(osp.join(save_path,'dev_label.txt'),'w')

test_data_file = open(osp.join(save_path,'test_data.txt'),'w')
test_label_file = open(osp.join(save_path,'test_label.txt'),'w')

num_train = 100
num_dev = 10
num_test = 10
num_imgs = 120

global_train_idx = 0
global_dev_idx = 0
global_test_idx = 0

for object in object_folders:
    if object == ".DS_Store":
        continue
    print(object)

    # load in tactile images and ground truth height maps
    tactile_path = osp.join(data_root_path,object,'gelsight_data','tactile_imgs')
    gt_heightmap_path = osp.join(data_root_path,object,'gelsight_data','gt_height_map')
    all_random_idx = np.random.permutation(num_imgs)
    train_idx = all_random_idx[0:num_train]
    dev_idx = all_random_idx[num_train:num_train+num_dev]
    test_idx = all_random_idx[num_train+num_dev:num_train+num_dev+num_test]
    for idx in train_idx:
        train_data_file.write(str(global_train_idx)+ " " + tactile_path+ "/" +str(idx)+".jpg"+ "\n")
        train_label_file.write(str(global_train_idx)+ " " + gt_heightmap_path+ "/" +str(idx)+".npy"+ "\n")
        global_train_idx += 1

    for idx in dev_idx:
        dev_data_file.write(str(global_dev_idx)+ " " + tactile_path + "/" +str(idx)+".jpg"+ "\n")
        dev_label_file.write(str(global_dev_idx)+ " " + gt_heightmap_path+ "/" +str(idx)+".npy"+ "\n")
        global_dev_idx += 1

    for idx in test_idx:
        test_data_file.write(str(global_test_idx)+ " " + tactile_path+ "/" +str(idx)+".jpg"+ "\n")
        test_label_file.write(str(global_test_idx)+ " " + gt_heightmap_path+ "/" +str(idx)+".npy"+ "\n")
        global_test_idx += 1

train_data_file.close()
train_label_file.close()
dev_data_file.close()
dev_label_file.close()
test_data_file.close()
test_label_file.close()
