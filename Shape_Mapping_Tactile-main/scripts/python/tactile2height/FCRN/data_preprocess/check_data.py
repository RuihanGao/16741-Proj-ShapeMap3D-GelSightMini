import numpy as np
import os
from os import path as osp
import shutil
np.random.seed(1)

# data_root_path = osp.join('..','..','..','dataset')
data_root_path = osp.join('/Users','ZilinSi','Desktop','shape_mapping','github', 'Shape_Mapping_Tactile', 'gelsight_data','textured_60sampled') # main data
object_folders = sorted(os.listdir(data_root_path))

for object in object_folders:
    if object == ".DS_Store":
        continue
    print(object)
    tactile_path = osp.join(data_root_path,object,'tactile_imgs')
    gt_heightmap_path = osp.join(data_root_path,object,'gt_height_map')
    gt_contact_mask_path = osp.join(data_root_path,object,'gt_contact_mask')
    tactile_folders = sorted(os.listdir(tactile_path))
    print(len(tactile_folders))
    heightmap_folders = sorted(os.listdir(gt_heightmap_path))
    print(len(heightmap_folders))
    mask_folders = sorted(os.listdir(gt_contact_mask_path))
    print(len(mask_folders))
