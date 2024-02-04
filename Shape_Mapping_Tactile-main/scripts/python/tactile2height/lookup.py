import os
from os import path as osp
import cv2
import numpy as np
from scipy.fftpack import dst, idst
from scipy.ndimage import gaussian_filter

import sys
sys.path.append("..")
import basics.params as pr
import basics.sensorParams as psp
from basics.CalibData import CalibData
# from config import lookup_table_config


class Simulation:
    def __init__(self,**config):
        self.calib_data = CalibData(config['path2model'])
        extension = osp.splitext(config['path2background'])[1]
        if (extension == '.npy'):
            self.bg_proc = np.load(osp.join(config['path2background']))
        else:
            self.bg_proc = cv2.imread(config['path2background'])

        self.data_folder = config['path2data']
        self.gelpad = np.load(config['path2gel_model'])
        self.save_folder = config['path2save']
        self.indent_depth = config['indent_depth']
        self.num_data = config['num_data']

    def simulate(self, object=None):
        if object is None:
            # generate height maps for all objects
            object_folders = sorted(os.listdir(self.data_folder))
        else:
            object_folders = object

        for obj in object_folders:
            if obj == ".DS_Store":
                continue
            print(obj)
            data_folder = self.data_folder + obj + '/tactile_imgs/'
            if not os.path.exists(data_folder):
                data_folder = self.data_folder + obj + '/gelsight/'
            save_folder = self.save_folder + obj + '/'
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            scale_gel = self.indent_depth/psp.pixmm
            max_gel = self.indent_depth/psp.pixmm
            # print("scale for gelpad is " + str(scale_gel))

            tactileImageFiles = (os.listdir(data_folder))

            for idx in range(self.num_data):
                img_path = data_folder + str(idx) + ".jpg"
                if not os.path.exists(img_path):
                    img_path = os.path.join(data_folder, tactileImageFiles[idx])

                img = cv2.imread(img_path)
                img = img.astype(int)
                self.bg_proc = self.bg_proc.astype(int)
                height_map = self.generate_single_height_map(img)
                scale_est = np.max(height_map) - np.min(height_map)
                if idx == 0:
                    scale = scale_gel/scale_est
                height_map = height_map * scale + max_gel - np.max(height_map) * scale
                np.save(save_folder + str(idx) + '.npy', height_map)

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
        # rgb2 = rgb1
        # rgb2[:,0] = (rgb1[:,0] - psp.zeropoint_r)/psp.lookscale_r
        # rgb2[:,1] = (rgb1[:,1] - psp.zeropoint_g)/psp.lookscale_g
        # rgb2[:,2] = (rgb1[:,2] - psp.zeropoint_b)/psp.lookscale_b
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
    sim = Simulation(**lookup_table_config)
    # obj = '002_master_chef_can'
    # sim.simulate(obj)
    sim.simulate()
