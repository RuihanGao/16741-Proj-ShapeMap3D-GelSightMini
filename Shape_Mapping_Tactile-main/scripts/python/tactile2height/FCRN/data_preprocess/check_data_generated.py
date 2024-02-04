import numpy as np
import os
from os import path as osp
import shutil
np.random.seed(1)

model = 'lookup' #'mlp' # 'contact_masks'
data_root_path = osp.join('/Users','ZilinSi','Desktop','shape_mapping','github', 'Shape_Mapping_Tactile', 'generated_data','textured_60sampled', model) # main data
object_folders = sorted(os.listdir(data_root_path))

for object in object_folders:
    if object == ".DS_Store":
        continue
    print(object)
    tactile_path = osp.join(data_root_path,object)
    tactile_folders = sorted(os.listdir(tactile_path))
    print(len(tactile_folders))
