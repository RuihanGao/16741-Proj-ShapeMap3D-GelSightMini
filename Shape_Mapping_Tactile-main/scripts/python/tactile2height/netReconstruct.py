import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from os import path as osp
import os
import cv2
import matplotlib.pyplot as plt
from scipy.fftpack import dst, idst
from scipy.ndimage import gaussian_filter

import sys
sys.path.append("..")
import basics.params as pr
import basics.sensorParams as psp
from basics.CalibData import CalibData

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

# 00
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
    def __init__(self, calibfolder):
        # print("setting devices...")
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if use_cuda else "cpu")
        # print(self.device)

        # print("setting parameters...")
        self.params = {'batch_size': 1024,
                'shuffle': False,
                'num_workers':6}
        calib = osp.join(calibfolder, 'mlp_net.pt')

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
    def __init__(self, obj, tactile_folder, calib_folder, save_folder):
        self.gt_folder =  osp.join(tactile_folder, 'gt_height_map')
        self.data_folder = osp.join(tactile_folder, 'tactile_imgs')
        self.mask_folder = osp.join(tactile_folder, 'gt_contact_mask')
        self.save_folder = save_folder
        self.calib_folder = calib_folder
        self.obj = obj

        rawData = osp.join(self.calib_folder, "dataPack.npz")
        data_file = np.load(rawData,allow_pickle=True)
        self.f0 = data_file['f0']
        self.bg_proc = self.processInitialFrame()
        self.image2heightmap = Model(self.calib_folder)

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

    def simulate(self):
        gel_map_path = osp.join(self.calib_folder, 'gelmap2.npy')
        gelpad = np.load(gel_map_path)
        maximum = np.percentile(gelpad, 95)
        minimum = np.percentile(gelpad, 5)
        scale_gel = 1.0/psp.pixmm
        max_gel = 1.0/psp.pixmm

        all_err = 0.0
        all_contact_err = 0.0

        N = len(os.listdir(self.data_folder)) # number of images

        for idx in range(N):
            img = cv2.imread(osp.join(self.data_folder, str(idx) + '.jpg'))
            gt = np.load(osp.join(self.gt_folder, str(idx) + '.npy'))
            gt_mask = np.load(osp.join(self.mask_folder, str(idx) + '.npy'))
            # print(img.shape)
            # plt.imshow(img)
            # plt.show()
            # p = input("pause")

            img = img.astype(int)
            self.bg_proc = self.bg_proc.astype(int)
            height_map = self.generate_single_height_map(img) # heightmap from image
            # height_map = -1 * height_map
            # plt.imshow(height_map)
            # plt.show()
            # p = input("pause")
            ######### test ##########
            if idx == 0:
                scale_gt = np.max(gt) - np.min(gt)
                scale_est = np.max(height_map) - np.min(height_map)
                # scale = scale_gt/scale_est # use ground truth scale
                scale = scale_gel/scale_est # use estimated scale
            height_map = height_map * scale + max_gel - np.max(height_map) * scale # pix output

            flip_height_map = -1 * (height_map - max_gel) * psp.pixmm
            flip_gt = -1 * (gt - max_gel) * psp.pixmm

            # compute error
            error = (flip_gt - flip_height_map)
            mean_error = np.mean(np.abs(error))
            all_err += mean_error
            mean_error_contact = np.mean(np.abs(error[gt_mask]))
            all_contact_err += mean_error_contact

            message = f"Error: {mean_error:.2f} mm, Contact error: {mean_error_contact:.2f} mm"
            # print(message)
            flip_height_map = np.multiply(flip_height_map, gt_mask)
            flip_gt = np.multiply(flip_gt, gt_mask)

            flip_gt[flip_gt==0] = np.nan
            flip_height_map[flip_height_map==0] = np.nan

            # ###### visualization ######
            # fig = plt.figure(1)
            # ax1 = fig.add_subplot(321)
            # ax1.title.set_text('Estimated height map')
            # plt.imshow(flip_height_map)

            # ax2 = fig.add_subplot(323)
            # ax2.title.set_text('Ground truth height map')
            # plt.imshow(flip_gt)

            # ax3 = fig.add_subplot(325)
            # ax3.title.set_text('Error map')
            # plt.imshow(np.abs(error))

            # ax = fig.add_subplot(1, 2, 2, projection='3d')
            # ax.title.set_text(message)
            # [xq, yq] = np.meshgrid(range(0, psp.w, 1), range(0, psp.h, 1))
            # # plt.gca().set_zlim(bottom=0.2)
            # ax.scatter(xq[::5,::5], yq[::5,::5], flip_height_map[::5,::5]  , marker='.', color='r', label='Estimated height map') # flip and convert to mm
            # ax.scatter(xq[::5,::5], yq[::5,::5], flip_gt[::5,::5] , marker='.', color='b', label='Ground truth height map') # flip and convert to mm
            # ax.set_xlabel('X Label')
            # ax.set_ylabel('Y Label')
            # ax.set_zlabel('Z Label')
            # ax.legend()
            # plt.show()

            ###### visualization end ######
            # p = input("pause")
            if not os.path.isdir(self.save_folder):
                os.makedirs(self.save_folder, exist_ok=True)

            np.save(osp.join(self.save_folder, str(idx) + '.npy'), height_map)
        all_err = all_err/N
        all_contact_err = all_contact_err/N
        message = f"All Error: {all_err:.2f} mm, All contact error: {all_contact_err:.2f} mm"

        filename = osp.join(self.save_folder, '..', 'error.txt')
        if os.path.exists(filename):
            append_write = 'a' # append if already exists
        else:
            append_write = 'w' # make a new file if not

        fopen = open(filename,append_write)
        fopen.write(self.obj + ', ' + f"{all_contact_err:.2f}" + '\n')
        fopen.close()
        print(message)

    def generate_single_height_map(self, img):
        dI = img - self.bg_proc # remove bg image
        im_grad_x, im_grad_y, im_grad_mag, im_grad_dir = self.match_grad(dI) # Image --> [Gx, Gy]
        height_map = self.fast_poisson(im_grad_x, im_grad_y)*psp.pixmm # [Gx, Gy] --> heightmap (mm)
        return height_map

    def vizGradient(self, mag, dir):
        plt.figure(1)
        plt.subplot(211)
        plt.imshow(mag)
        plt.subplot(212)
        plt.imshow(dir)
        plt.show()
        return

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

    def fast_poisson(self, gx, gy):
        [h, w] = gx.shape
        gxx = np.zeros((h,w))
        gyy = np.zeros((h,w))

        j = np.arange(h-1)
        k = np.arange(w-1)

        [tmpx, tmpy] = np.meshgrid(j,k)
        # gradient of gradient
        gyy[tmpx+1, tmpy] = gy[tmpx+1, tmpy] - gy[tmpx,tmpy];
        gxx[tmpx,tmpy+1] = gx[tmpx, tmpy+1] - gx[tmpx, tmpy];

        f = gxx + gyy
        f2 = f[1:-1, 1:-1] # ignore edges

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
    if len(sys.argv) == 3:
        obj = str(sys.argv[1]) # 'mustard_bottle', 'power_drill'
        touchFile = str(sys.argv[2]) # 'mustard_bottle', 'power_drill'
    else:
        obj = "005_tomato_soup_can"
        touchFile = "textured_60sampled.mat"

    # change path to cwd
    abspath = osp.abspath(__file__)
    dname = osp.dirname(abspath)
    os.chdir(dname)

    print(obj)
    tactile_folder = osp.join('..','..','..','gelsight_data', os.path.splitext(touchFile)[0], obj)
    # TODO: make for the lookup/MLP models
    save_folder = osp.join('..','..','..', 'generated_data', os.path.splitext(touchFile)[0], 'mlp', obj)
    calib_folder = osp.join('..','..','..','calib')
    sim = Simulation(obj, tactile_folder, calib_folder, save_folder)
    sim.simulate()
