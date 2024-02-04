import os
from os import path as osp
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.fftpack import dst, idst
from scipy.ndimage import gaussian_filter

import sys
sys.path.append("..")
import basics.params as pr
import basics.sensorParams as psp
from basics.CalibData import CalibData
# from config import mlp_net_config

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data):
        # data: N * 5 (r, g, b, x, y)
        # label: N * 2 (grad_mag, grad_dir)
        self.data = data
        self.length = len(self.data)
    def __len__(self):
        return self.length
    def __getitem__(self, index):
        return self.data[index]

class MLPNet(nn.Module):
    def __init__(self):
        super(MLPNet,self).__init__()
        self.dropout = nn.Dropout(0.30)
        self.fc1 = nn.Linear(5,32)
        self.bn1 = nn.BatchNorm1d(32)
        self.fc2 = nn.Linear(32,32)
        self.bn2 = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(32,32)
        self.bn3 = nn.BatchNorm1d(32)
        self.fc4 = nn.Linear(32,2)
        # self.dropout = nn.Dropout(0.30)
        # self.fc1 = nn.Linear(5,16)
        # self.bn1 = nn.BatchNorm1d(16)
        # self.fc2 = nn.Linear(16,32)
        # self.bn2 = nn.BatchNorm1d(32)
        # self.fc3 = nn.Linear(32,16)
        # self.bn3 = nn.BatchNorm1d(16)
        # self.fc4 = nn.Linear(16,2)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = self.fc4(x)
        return x

class Model:
    def __init__(self, **config):
        # print("setting devices...")
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if use_cuda else "cpu")
        # print(self.device)

        # print("setting parameters...")
        self.params = {'batch_size': 1024,
                'shuffle': False,
                'num_workers':6}
        calib = config['path2model']

        self.model = MLPNet()
        self.model.load_state_dict(torch.load(calib))
        self.model.eval()

    def test(self, test_data):
        # test_data: [640*480, 5]
        # result: [640*480, 2]

        test_set = Dataset(test_data)
        test_loader = torch.utils.data.DataLoader(test_set, **self.params)
        alloutput = []
        with torch.no_grad():
            for data in test_loader:
                data = data.to(self.device)
                output = self.model(data.float()).numpy()
                alloutput.append(output)
        result = np.concatenate(alloutput,axis=0)
        return result

class Simulation:
    def __init__(self,**config):
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
        self.image2heightmap = Model(**config)

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

    def magDir2GxGy(self, mag, dir):
        tmp = np.tan(mag)
        Gx = tmp*np.cos(dir)
        Gy = tmp*np.sin(dir)
        return Gx, Gy

    def match_grad(self, dI):
        h, w = dI.shape[:2]
        [xqq, yqq] = np.meshgrid(range(w), range(h))
        rv = dI[:,:,0]
        gv = dI[:,:,1]
        bv = dI[:,:,2]

        # separate to the RGB components
        rf = rv.reshape(h*w)
        gf = gv.reshape(h*w)
        bf = bv.reshape(h*w)

        xf = xqq.reshape(h*w)
        yf = yqq.reshape(h*w)
        test_data = np.stack([rf,gf,bf,xf,yf],axis=1) # RGBXY = [640*480, 5]

        # Network output mag, dir [640*480, 2]
        output = self.image2heightmap.test(test_data)

        im_grad_mag = output[:,0].reshape((h,w))
        im_grad_dir = output[:,1].reshape((h,w))

        # visualize gradient mag and direction
        # self.vizGradient(im_grad_mag, im_grad_dir)

        # split gradient to [Gx,Gy]
        im_grad_x, im_grad_y = self.magDir2GxGy(im_grad_mag, im_grad_dir)
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
    sim = Simulation(**mlp_net_config)
    # obj = '002_master_chef_can'
    # sim.simulate(obj)
    sim.simulate()
