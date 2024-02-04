import numpy as np
import os
from os import path as osp
import shutil
np.random.seed(1)

# data_root_path = osp.join('..','..','..','dataset')
data_root_path = osp.join('/Users','ZilinSi','Desktop','shape_mapping','github', 'Shape_Mapping_Tactile', 'models') # main data
object_folders = sorted(os.listdir(data_root_path))

for object in object_folders:
    if object == ".DS_Store":
        continue
    print(object)

    txt_mat = 'textured.mat'
    txt_50_mat = 'textured_50sampled.mat'
    txt_120_mat = 'textured_120sampled.mat'
    txt_50_pdf = 'textured_50sampled.pdf'
    txt_120_pdf = 'textured_120sampled.pdf'
    google_512k = 'google_512k'

    items = sorted(os.listdir(osp.join(data_root_path,object)))
    print(len(items))
    google_path = osp.join(data_root_path,object,'google_512k')
    google_folders = sorted(os.listdir(google_path))
    print(len(google_folders))
