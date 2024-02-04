import os
from os import path as osp
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dst, idst
from scipy.ndimage import gaussian_filter

import sys
sys.path.append("..")
import basics.params as pr
import basics.sensorParams as psp
from basics.CalibData import CalibData
from real_config import contact_mask_config


class Simulation:
    def __init__(self,**config):
        # extension = osp.splitext(config['path2background'])[1]
        # if (extension == '.npy'):
        #     self.bg_proc = np.load(osp.join(config['path2background']))
        # else:
        #     self.bg_proc = cv2.imread(config['path2background'])
        self.data_folder = config['path2data']
        self.height_folder = config['path2sim_height_map']
        self.gelpad = np.load(config['path2gel_model'])
        self.save_folder = config['path2save']
        self.num_data = config['num_data']
        self.data_source = config['data_source']

        [self.h,self.w] = self.gelpad.shape
        # max_g = np.max(self.gelpad)
        # self.gelpad = -1*(self.gelpad - max_g)

        self.init_height = self.gelpad

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
            data_folder = self.height_folder + obj + '/'
            gt_folder = self.data_folder + obj + '/gt_contact_mask/'
            gt_height_folder = self.data_folder + obj + '/gt_height_map/'
            save_folder = self.save_folder + obj + '/'
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)

            for idx in range(self.num_data):
                height_map = np.load(data_folder + str(idx) + '.npy')
                dim = (self.w, self.h)
                resized_height = cv2.resize(height_map, dim, interpolation = cv2.INTER_AREA)
                gt = np.load(gt_folder + str(idx) + '.npy')
                gt_height = np.load(gt_height_folder + str(idx) + '.npy')
                contact_mask = self.generate_single_contack_mask_from_height(resized_height)
                plt.figure(1)
                plt.subplot(221)
                plt.imshow(resized_height)

                plt.subplot(222)
                plt.imshow(gt_height)

                plt.subplot(223)
                plt.imshow(self.init_height)

                plt.subplot(224)
                plt.imshow(contact_mask)
                plt.show()

                # plt.subplot(142)
                # plt.imshow(self.init_height)
                # #

                # plt.subplot(143)
                # plt.imshow(diff_heights)

                # plt.subplot(144)
                # plt.imshow(contact_mask)

                # plt.show(block=False)
                # p = input("pause")

                np.save(save_folder + str(idx) + '.npy', contact_mask)

    def generate_single_contack_mask_from_height(self, cur_height):
        if self.data_source == 'real':
            cur_height = cur_height[100:-100,100:-100]
            init_height = self.init_height[100:-100,100:-100]
            diff_heights = cur_height - init_height
            diff_heights[diff_heights<10]=0
            contact_mask = diff_heights > np.percentile(diff_heights, 90)*0.50 #*0.8

            padded_contact_mask = np.zeros(self.init_height.shape, dtype=bool)
            padded_contact_mask[100:-100,100:-100] = contact_mask
        else:
            cur_height = cur_height[20:-20,20:-20]
            init_height = self.init_height[20:-20,20:-20]
            diff_heights = cur_height - init_height
            diff_heights[diff_heights<5]=0
            contact_mask = diff_heights > np.percentile(diff_heights, 90)*0.50 #*0.8
            
            padded_contact_mask = np.zeros(self.init_height.shape, dtype=bool)
            padded_contact_mask[20:-20,20:-20] = contact_mask

        return padded_contact_mask, diff_heights

    def generate_single_contack_mask(self, img):
        diffim = np.abs(np.int16(img) - self.bg_proc)
        plt.imshow(diffim)
        plt.show()
        # maxim=diffim.max(axis=2)
        # max_img = np.amax(diffim,2)
        max_img = np.sum(diffim,2)
        plt.imshow(max_img)
        plt.show()
        contactmap=max_img
        countnum=(contactmap>20).sum()
        # print(countnum)

        contactmap[contactmap<20]=0
        # contactmap[contactmap<=0]=0

        image = np.zeros((480,640))

        maxC = np.max(contactmap)
        sec90C = np.percentile(contactmap, 90)
        sec95C = np.percentile(contactmap, 95)
        sec99C = np.percentile(contactmap, 99)


        contact_mask = contactmap>0.4*sec90C

        total_contact = np.sum(contact_mask)
        self.contact_ratio = total_contact/(image.shape[0]*image.shape[1])
        # print("contact ratio is " + str(self.contact_ratio))

        return contact_mask



if __name__ == "__main__":
    sim = Simulation(**contact_mask_config)
    obj = ["021_bleach_cleanser"] # 002_master_chef_can # 011_banana
    # sim.simulate(obj)
    sim.simulate(obj)
