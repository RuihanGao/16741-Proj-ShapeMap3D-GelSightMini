import gc
from glob import glob
from os import path as osp
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.linalg import lstsq
from scipy.ndimage import gaussian_filter
from scipy import interpolate

import sys
sys.path.append("..")
# from utils import find_ball_params, lookuptable_from_ball, lookuptable_smooth
import basics.params as pr
import basics.sensorParams as psp
from basics.Geometry import Circle

from ipdb import set_trace

class rawData:
    # store r, g, b, x, y
    rv = None
    gv = None
    bv = None
    xv = None
    yv = None
    grad_mag = None
    grad_dir = None

class ReconstructCalibration:
    def __init__(self,fn):
        self.fn = osp.join(fn, "dataPack.npz")
        data_file = np.load(self.fn,allow_pickle=True)

        self.f0 = data_file['f0']
        self.BallRad = psp.ball_radius
        self.Pixmm = psp.pixmm
        self.imgs = data_file['imgs']
        self.radius_record = data_file['touch_radius']
        self.touchCenter_record = data_file['touch_center']

        self.bg_proc = self.processInitialFrame()
        self.raw_data = rawData()

        self.img_data_dir = fn

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
    def calibrate_all(self):
        num_img = np.shape(self.imgs)[0]
        for idx in range(num_img):
            print("# iter " + str(idx))
            self.calibrate_single(idx)
            pause = input("pause here")
        out_fn_path = osp.join(self.img_data_dir, "reconstruct_raw.npz")
        np.savez(out_fn_path,\
            rv=self.raw_data.rv,
            gv=self.raw_data.gv,
            bv=self.raw_data.bv,
            xv=self.raw_data.xv,
            yv=self.raw_data.yv,
            grad_mag = self.raw_data.grad_mag,
            grad_dir = self.raw_data.grad_dir)
        print("Saved!")

    def calibrate_single(self, idx):
        bins = psp.numBins
        ball_radius_pix = psp.ball_radius/psp.pixmm

        circle = Circle(int(self.touchCenter_record[idx,0]), int(self.touchCenter_record[idx,1]),int(self.radius_record[idx]))
        center = circle.center
        radius = circle.radius

        frame = self.imgs[idx,:,:,:]
        dI = frame.astype("float") - self.bg_proc
        print(np.min(dI))
        print(np.max(dI))

        sizey, sizex = dI.shape[:2]
        [xqq, yqq] = np.meshgrid(range(sizex), range(sizey))
        xq = xqq - center[0]
        yq = yqq - center[1]

        rsqcoord = xq*xq + yq*yq
        rad_sq = radius*radius

        # print(rad_sq)
        # print(int(ball_radius_pix*ball_radius_pix))
        valid_rad = min(rad_sq, int(ball_radius_pix*ball_radius_pix))
        valid_mask = rsqcoord < (valid_rad)

        validId = np.nonzero(valid_mask)
        xvalid = xq[validId]; yvalid = yq[validId]
        rvalid = np.sqrt( xvalid*xvalid + yvalid*yvalid)

        if( np.max(rvalid - ball_radius_pix) > 0):
            print("Contact Radius(%f) is too large(%f). Ignoring the exceeding area"%(np.max(rvalid), ball_radius_pix))
            rvalid[rvalid>ball_radius_pix] = ball_radius_pix - 0.001

        gradxseq = np.arcsin(rvalid/ball_radius_pix)
        gradyseq = np.arctan2(-yvalid, -xvalid)

        rv = dI[:,:,0][validId]
        gv = dI[:,:,1][validId]
        bv = dI[:,:,2][validId]

        xv = xqq[validId]
        yv = yqq[validId]
        if idx == 0:
            self.raw_data.rv = rv
            self.raw_data.gv = gv
            self.raw_data.bv = bv
            self.raw_data.xv = xv
            self.raw_data.yv = yv
            self.raw_data.grad_mag = gradxseq
            self.raw_data.grad_dir = gradyseq
        else:
            self.raw_data.rv = np.concatenate((self.raw_data.rv, rv), axis=0)
            self.raw_data.gv = np.concatenate((self.raw_data.gv, gv), axis=0)
            self.raw_data.bv = np.concatenate((self.raw_data.bv, bv), axis=0)
            self.raw_data.xv = np.concatenate((self.raw_data.xv, xv), axis=0)
            self.raw_data.yv = np.concatenate((self.raw_data.yv, yv), axis=0)
            self.raw_data.grad_mag = np.concatenate((self.raw_data.grad_mag, gradxseq), axis=0)
            self.raw_data.grad_dir = np.concatenate((self.raw_data.grad_dir, gradyseq), axis=0)



if __name__ == "__main__":
    # data_folder = osp.join('..', '..', 'data', 'N4.00mm')
    data_folder = osp.join(osp.join("..", "..", "..", "calib"))
    polyCalib = ReconstructCalibration(data_folder)
    polyCalib.calibrate_all()
