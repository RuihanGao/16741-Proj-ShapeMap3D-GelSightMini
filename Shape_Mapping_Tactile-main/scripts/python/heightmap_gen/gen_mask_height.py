import os
from os import path as osp
import cv2
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("..")
from basics.CalibData import CalibData
from config import mask_height_config


class Simulation:
    def __init__(self,**config):
        # self.bg_proc = np.load(osp.join(config['path2background']))
        self.data_folder = config['path2data']
        self.save_folder = config['path2save']
        self.num_data = config['num_data']

    def simulate(self, object=None):
        if object is None:
            # generate height maps for all objects
            object_folders = sorted(os.listdir(self.data_folder))
        else:
            object_folders = [object]

        for obj in object_folders:
            if obj == ".DS_Store":
                continue
            print(obj)
            gt_folder = self.data_folder + obj + '/gt_contact_mask/'
            gt_height_folder = self.data_folder + obj + '/gt_height_map/'
            save_folder = self.data_folder + obj + '/gt_mask_geights/'
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)

            for idx in range(self.num_data):
                gt_height = np.load(gt_height_folder + str(idx) + '.npy')
                gt_mask = np.load(gt_folder + str(idx) + '.npy')
                gt_mask_height = gt_mask * gt_height
                plt.imshow(gt_mask_height)
                plt.show()
                np.save(save_folder + str(idx) + '.npy', gt_mask_height)



if __name__ == "__main__":
    sim = Simulation(**mask_height_config)
    obj = '042_adjustable_wrench' # 002_master_chef_can # 011_banana
    sim.simulate(obj)
    # sim.simulate()
