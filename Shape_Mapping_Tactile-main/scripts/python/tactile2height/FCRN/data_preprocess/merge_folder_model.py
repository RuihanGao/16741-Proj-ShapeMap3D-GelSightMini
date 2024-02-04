import numpy as np
import os
from os import path as osp
import shutil
np.random.seed(1)

# data_root_path = osp.join('..','..','..','dataset')
data_root_path = osp.join('/Users','ZilinSi','Desktop','shape_mapping','github', 'Shape_Mapping_Tactile', 'models') # main data
object_folders = sorted(os.listdir(data_root_path))

merge_data_path = osp.join('/Users','ZilinSi','Downloads','models 4') # main data
merge_folders = sorted(os.listdir(merge_data_path))

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


    cur = osp.join(merge_data_path,object,txt_mat)
    to = osp.join(data_root_path,object,txt_mat)
    if os.path.isfile(cur):
        shutil.move(cur, to)

    cur = osp.join(merge_data_path,object,txt_50_mat)
    to = osp.join(data_root_path,object,txt_50_mat)
    if os.path.isfile(cur):
        shutil.move(cur, to)

    cur = osp.join(merge_data_path,object,txt_120_mat)
    to = osp.join(data_root_path,object,txt_120_mat)
    if os.path.isfile(cur):
        shutil.move(cur, to)

    cur = osp.join(merge_data_path,object,txt_50_pdf)
    to = osp.join(data_root_path,object,txt_50_pdf)
    if os.path.isfile(cur):
        shutil.move(cur, to)

    cur = osp.join(merge_data_path,object,txt_120_pdf)
    to = osp.join(data_root_path,object,txt_120_pdf)
    if os.path.isfile(cur):
        shutil.move(cur, to)

    google_path = osp.join(data_root_path, object, google_512k)
    m_google_path = osp.join(merge_data_path, object, google_512k)
    if os.path.isdir(m_google_path):
        m_google_folders = sorted(os.listdir(m_google_path))
        if not os.path.isdir(google_path):
            os.makedirs(google_path)
        for tactile in m_google_folders:
            cur = osp.join(merge_data_path, object, google_512k,tactile)
            to = osp.join(data_root_path, object, google_512k, tactile)
            shutil.move(cur, to)


    # tactile_path = osp.join(data_root_path,object,'tactile_imgs')
    # gt_heightmap_path = osp.join(data_root_path,object,'gt_height_map')
    # gt_contact_mask_path = osp.join(data_root_path,object,'gt_contact_mask')
    #
    # m_tactile_path = osp.join(merge_data_path,object,'tactile_imgs')
    # m_gt_heightmap_path = osp.join(merge_data_path,object,'gt_height_map')
    # m_gt_contact_mask_path = osp.join(merge_data_path,object,'gt_contact_mask')
    # if os.path.isdir(m_tactile_path):
    #     m_tactile_folders = sorted(os.listdir(m_tactile_path))
    #     for tactile in m_tactile_folders:
    #         cur = osp.join(merge_data_path,object,'tactile_imgs',tactile)
    #         to = osp.join(data_root_path,object,'tactile_imgs', tactile)
    #         shutil.move(cur, to)
    #
    # if os.path.isdir(m_gt_heightmap_path):
    #     m_gt_folder = sorted(os.listdir(m_gt_heightmap_path))
    #     for gt in m_gt_folder:
    #         cur = osp.join(merge_data_path,object,'gt_height_map',gt)
    #         to = osp.join(data_root_path,object,'gt_height_map', gt)
    #         shutil.move(cur, to)
    # if os.path.isdir(m_gt_contact_mask_path):
    #     m_mask_folder = sorted(os.listdir(m_gt_contact_mask_path))
    #     for mask in m_mask_folder:
    #         cur = osp.join(merge_data_path,object,'gt_contact_mask',mask)
    #         to = osp.join(data_root_path,object,'gt_contact_mask', mask)
    #         shutil.move(cur, to)
