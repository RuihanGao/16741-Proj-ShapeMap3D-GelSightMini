import numpy as np
import os
from os import path as osp
import shutil
np.random.seed(1)

model = 'lookup' # 'fcrn' # 'contact_masks'
data_root_path = osp.join('/Users','ZilinSi','Desktop','shape_mapping', 'github', 'Shape_Mapping_Tactile','generated_data','textured_60sampled', model) # main data
object_folders = sorted(os.listdir(data_root_path))

merge_data_path = osp.join('/Users','ZilinSi','Downloads','lookup 3') # to be merged data
merge_folders = sorted(os.listdir(merge_data_path))

for object in object_folders:
    if object == ".DS_Store":
        continue
    print(object)

    tactile_path = osp.join(data_root_path,object)

    m_tactile_path = osp.join(merge_data_path,object)
    if os.path.isdir(m_tactile_path):
        m_tactile_folders = sorted(os.listdir(m_tactile_path))
        for tactile in m_tactile_folders:
            cur = osp.join(merge_data_path,object,tactile)
            to = osp.join(data_root_path,object,tactile)
            shutil.move(cur, to)
