import gc
from glob import glob
from os import path as osp
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dst, idst
from scipy.ndimage import gaussian_filter
from scipy.ndimage import correlate
from PIL import Image, ImageFilter

import sys
sys.path.append("..")
# from utils import simulate_img
import basics.params as pr
import basics.sensorParams as psp
from basics.Geometry import Circle
from basics.CalibData import CalibData

from ipdb import set_trace


class Simulation:
    def __init__(self,fn):
        calib_data = osp.join(fn, "reconstruct_calib.npz")
        self.calib_data = CalibData(calib_data)

        rawData = osp.join(fn, "dataPack.npz")
        data_file = np.load(rawData,allow_pickle=True)
        self.f0 = data_file['f0']
        self.bg_proc = self.processInitialFrame()

    def processInitialFrame(self):
        # gaussian filtering with square kernel with
        # filterSize : kscale*2+1
        # sigma      : kscale
        kscale = pr.kscale

        img_d = self.f0.astype('float')
        convEachDim = lambda in_img :  gaussian_filter(in_img, kscale)

        f0 = self.f0.copy()
        for ch in range(img_d.shape[2]):
            f0[:,:, ch] = convEachDim(img_d[:,:,ch])

        frame_ = img_d

        # Checking the difference between original and filtered image
        diff_threshold = pr.diffThreshold
        dI = np.mean(f0-frame_, axis=2)
        idx =  np.nonzero(dI<diff_threshold)

        # Mixing image based on the difference between original and filtered image
        frame_mixing_per = pr.frameMixingPercentage
        h,w,ch = f0.shape
        pixcount = h*w

        for ch in range(f0.shape[2]):
            f0[:,:,ch][idx] = frame_mixing_per*f0[:,:,ch][idx] + (1-frame_mixing_per)*frame_[:,:,ch][idx]

        return f0
    def simulate(self, tactile_folder):
        gt_folder = tactile_folder + 'gt_height_map/'
        data_folder = tactile_folder + 'tactile_imgs/'
        mask_folder = tactile_folder + 'gt_contact_mask/'
        save_folder = tactile_folder + 'generated_height_map/'
        gel_map_path = osp.join('..','config', 'gelmap2.npy')

        gelpad = np.load(gel_map_path)
        maximum = np.percentile(gelpad, 95)
        minimum = np.percentile(gelpad, 5)
        print(maximum,minimum)
        # scale_gel = np.max(gelpad) - np.min(gelpad)
        scale_gel = 1.5/psp.pixmm
        # max_gel = np.max(gelpad)
        max_gel = 1.5/psp.pixmm
        print("scale for gelpad is " + str(scale_gel))
        # plt.imshow(gelpad)
        # plt.show()
        # p = input("pause")
        all_err = 0.0
        all_contact_err = 0.0
        for idx in range(50):
            img = cv2.imread(data_folder + str(idx) + '.jpg')
            gt = np.load(gt_folder + str(idx) + '.npy')
            gt_mask = np.load(mask_folder + str(idx) + '.npy')
            # print(img.shape)
            # plt.imshow(img)
            # plt.show()
            # p = input("pause")

            img = img.astype(int)
            self.bg_proc = self.bg_proc.astype(int)
            height_map = self.generate_single_height_map(img)
            # height_map = -1 * height_map
            # plt.imshow(height_map)
            # plt.show()
            ######### test ##########
            # if idx == 0:
            scale_gt = np.max(gt) - np.min(gt)
            scale_est = np.max(height_map) - np.min(height_map)
            # scale = scale_gt/scale_est
            if idx == 0:
                scale = scale_gel/scale_est
            height_map = height_map * scale + max_gel - np.max(height_map) * scale
            print("ground truth")
            print(np.min(gt),np.max(gt))
            print("estimated")
            print(np.min(height_map),np.max(height_map))
            error = gt - height_map
            print("error")
            print(np.mean(np.abs(error)))
            all_err += np.mean(np.abs(error))
            print("error contact")
            print(np.mean(np.abs(error[gt_mask])))
            all_contact_err += np.mean(np.abs(error[gt_mask]))
            # plt.figure(1)
            # plt.subplot(311)
            # plt.imshow(gt)
            #
            # plt.subplot(312)
            # plt.imshow(height_map)
            #
            # plt.subplot(313)
            # plt.imshow(np.abs(error))
            # plt.show()

            # plt.imshow(gt)
            # plt.show()
            # plt.imshow(height_map)
            # plt.show()
            # plt.imshow(np.abs(error))
            # plt.show()
            ######### test ##########

            # ###### visualization ######
            # fig = plt.figure()
            # ax = fig.add_subplot(projection='3d')
            # [xq, yq] = np.meshgrid(range(psp.w), range(psp.h))
            # ax.scatter(xq, yq, -1*height_map, marker='o')
            # ax.set_xlabel('X Label')
            # ax.set_ylabel('Y Label')
            # ax.set_zlabel('Z Label')
            # plt.show()
            ###### visualization end ######
            # p = input("pause")
            np.save(save_folder + str(idx) + '.npy', height_map)
        print("all error is ")
        print(all_err/50)
        print("all contact error is ")
        print(all_contact_err/50)

    def generate_single_height_map(self, img):
        dI = img - self.bg_proc
        im_grad_x, im_grad_y, im_grad_mag, im_grad_dir = self.match_grad(dI)
        height_map = self.fast_poisson(im_grad_x, im_grad_y)*psp.pixmm
        return height_map

    def match_grad(self, dI):
        f01 = self.bg_proc.copy()
        # f01[f01==0] = 1
        t = np.mean(f01)
        f01 = 1 + ((t/(self.bg_proc+1)) - 1)*2

        sizex, sizey = dI.shape[:2]

        im_grad_mag = np.zeros((sizex, sizey))
        im_grad_dir = np.zeros((sizex, sizey))

        ndI = dI*f01

        binm = self.calib_data.bins - 1
        rgb1 = ndI
        rgb2 = (rgb1 - psp.zeropoint)/psp.lookscale
        rgb2[rgb2>1] = 1
        rgb2[rgb2<0] = 0

        rgb3 = np.floor(rgb2*binm).astype('uint')
        r3 = rgb3[:,:,0]
        g3 = rgb3[:,:,1]
        b3 = rgb3[:,:,2]
        im_grad_mag = self.calib_data.grad_mag[r3, g3, b3]
        im_grad_dir = self.calib_data.grad_dir[r3, g3, b3]

        tmp = np.tan(im_grad_mag)
        im_grad_x = tmp*np.cos(im_grad_dir)
        im_grad_y = tmp*np.sin(im_grad_dir)

        return im_grad_x, im_grad_y, im_grad_mag, im_grad_dir
    def fast_poisson(self,gx, gy):
        [h, w] = gx.shape
        gxx = np.zeros((h,w))
        gyy = np.zeros((h,w))

        j = np.arange(h-1)
        k = np.arange(w-1)

        [tmpx, tmpy] = np.meshgrid(j,k)
        gyy[tmpx+1, tmpy] = gy[tmpx+1, tmpy] - gy[tmpx,tmpy];
        gxx[tmpx,tmpy+1] = gx[tmpx, tmpy+1] - gx[tmpx, tmpy];

        f = gxx + gyy

        f2 = f[1:-1, 1:-1]

        tt = dst(f2, type=1, axis=0)/2

        f2sin = (dst(tt.T, type=1, axis=0)/2).T

        [x,y] = np.meshgrid(np.arange(1, w-1), np.arange(1, h-1))

        denom = (2*np.cos(np.pi*x/(w-1))-2) + (2*np.cos(np.pi*y/(h-1)) - 2)

        f3 = f2sin/denom

        [xdim, ydim] = tt.shape
        tt = idst(f3, type=1, axis=0)/(xdim)
        img_tt = (idst(tt.T, type=1, axis=0)/ydim).T

        img_direct = np.zeros((h, w))
        img_direct[1:-1,1:-1] = img_tt

        return img_direct

if __name__ == "__main__":
    # data_folder = osp.join('..', '..', 'data', 'N4.00mm')
    data_folder = osp.join(osp.join( "..", "data", "04_26_2021","calib_ball"))
    sim = Simulation(data_folder)
    obj = '002_master_chef_can'
    tactile_folder = osp.join('..','..','..','..','data', obj, 'gelsight_data')

    sim.simulate(tactile_folder)
