import gc
from glob import glob
from os import path as osp
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.ndimage import gaussian_filter

import sys
sys.path.append("..")
# from utils import find_ball_params, lookuptable_from_ball, lookuptable_smooth
import basics.params as pr
import basics.sensorParams as psp
from basics.Geometry import Circle

class Grads:
  """each grad contains the N*N*params table for one (normal_mag, normal_dir) pair"""
  grad_mag = None
  grad_dir = None
  countmap = None

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
        self.grads = Grads()

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
            # pause = input("pause here")
        print("smooth")
        grad_mag, grad_dir = self.lookuptable_smooth()
        out_fn_path = osp.join(self.img_data_dir, "reconstruct_calib.npz")
        np.savez(out_fn_path,\
            bins=psp.numBins,
            grad_mag = grad_mag,
            grad_dir = grad_dir)
        print("Saved!")

    def calibrate_single(self, idx):
        bins = psp.numBins
        ball_radius_pix = psp.ball_radius/psp.pixmm

        zeropoint = psp.zeropoint
        lookscale = psp.lookscale

        circle = Circle(int(self.touchCenter_record[idx,0]), int(self.touchCenter_record[idx,1]),int(self.radius_record[idx]))
        center = circle.center
        radius = circle.radius

        frame = self.imgs[idx,:,:,:]
        dI = frame.astype("float") - self.bg_proc
        # plt.imshow(dI.astype(int))
        # plt.show()
        # p = input("pause")
        f01 = self.bg_proc.copy()
        f01[f01==0] = 1
        t = np.mean(f01)
        f01 = 1 + ((t/(self.bg_proc+1)) - 1)*2


        sizey, sizex = dI.shape[:2]
        [xq, yq] = np.meshgrid(range(sizex), range(sizey))
        xq = xq - center[0]
        yq = yq - center[1]

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

        binm = bins - 1

        r1 = dI[:,:,0][validId]*f01[:,:,0][validId]
        g1 = dI[:,:,1][validId]*f01[:,:,1][validId]
        b1 = dI[:,:,2][validId]*f01[:,:,2][validId]
        #
        # r1 = dI[:,:,0][validId]
        # g1 = dI[:,:,1][validId]
        # b1 = dI[:,:,2][validId]

        rgb1 = np.stack((r1, g1, b1), axis=1)
        print("max r: " + str(np.max(rgb1[:,0])))
        print("min r" + str(np.min(rgb1[:,0])))

        print("max g: " + str(np.max(rgb1[:,1])))
        print("min g" + str(np.min(rgb1[:,1])))

        print("max b: " + str(np.max(rgb1[:,2])))
        print("min b" + str(np.min(rgb1[:,2])))
        rgb2 = rgb1
        # rgb2[:,0] = (rgb1[:,0] - psp.zeropoint_r)/psp.lookscale_r
        # rgb2[:,1] = (rgb1[:,1] - psp.zeropoint_g)/psp.lookscale_g
        # rgb2[:,2] = (rgb1[:,2] - psp.zeropoint_b)/psp.lookscale_b

        rgb2 = (rgb1 - zeropoint)/lookscale
        rgb2[rgb2<0] = 0; rgb2[rgb2>1] = 1

        rgb3 = np.floor(rgb2*binm).astype('int')

        r3 = rgb3[:,0]
        g3 = rgb3[:,1]
        b3 = rgb3[:,2]

        # initialize for the first time
        if self.grads.grad_mag is None:
            self.grads.grad_mag = np.zeros((bins,bins,bins))
            self.grads.grad_dir = np.zeros((bins,bins,bins))
            self.grads.countmap = np.zeros((bins,bins,bins), dtype='uint8')

        tmp = self.grads.countmap[r3, g3, b3]
        self.grads.countmap[r3, g3, b3] = self.grads.countmap[r3, g3, b3] + 1

        # when the gradient is added the first time
        idx = np.where(tmp==0)
        self.grads.grad_mag[r3[idx], g3[idx], b3[idx]] = gradxseq[idx]
        self.grads.grad_dir[r3[idx], g3[idx], b3[idx]] = gradyseq[idx]

        # updating the gradients
        idx = np.where(tmp>0)
        self.grads.grad_mag[r3[idx], g3[idx], b3[idx]] = (self.grads.grad_mag[r3[idx], g3[idx], b3[idx]]*tmp[idx] + gradxseq[idx])/(tmp[idx]+1)

        # wrap around checks
        a1 = self.grads.grad_dir[r3[idx], g3[idx], b3[idx]]
        a2 = gradyseq[idx]

        diff_angle = a2 - a1
        a2[diff_angle > np.pi] -= 2*np.pi
        a2[-diff_angle > np.pi] += 2*np.pi

        self.grads.grad_dir[r3[idx], g3[idx], b3[idx]] = (a1*tmp[idx]+a2)/(tmp[idx]+1)


    def lookuptable_smooth(self):
        bins = psp.numBins
        countmap = self.grads.countmap
        grad_mag =  self.grads.grad_mag
        grad_dir = self.grads.grad_dir

        if not countmap[0,0,0] or countmap[0,0,0] == 1:

            grad_mag[0,0,0] == 0
            grad_dir[0,0,0] == 0

        validid = np.nonzero(countmap)
        print("# valid: " + str(validid[0].size))
        print("# all: " + str(bins**3))
        p = input("pause")

        # no interpolation needed
        if(validid[0].size == (bins**3)):
            return grad_mag, grad_dir

        # Nearest neighbor interpolation
        xout, yout, zout = np.meshgrid(range(bins), range(bins), range(bins))

        invalid_id = np.nonzero((countmap == 0))

        xvalid = xout[validid]; yvalid = yout[validid]; zvalid = zout[validid]
        xinvalid = xout[invalid_id]; yinvalid = yout[invalid_id]; zinvalid = zout[invalid_id]

        xyzvalid = np.stack((xvalid, yvalid, zvalid), axis=1)
        xyzinvalid = np.stack((xinvalid, yinvalid, zinvalid), axis=1)
        dist = cdist(xyzinvalid, xyzvalid)
        closest_id = np.argmin(dist, axis=1)
        closest_valid_idx = (validid[0][closest_id], validid[1][closest_id], validid[2][closest_id])
        grad_mag[invalid_id] = grad_mag[closest_valid_idx]
        grad_dir[invalid_id] = grad_dir[closest_valid_idx]
        return grad_mag, grad_dir

if __name__ == "__main__":
    # data_folder = osp.join('..', '..', 'data', 'N4.00mm')
    data_folder = osp.join(osp.join( "..", "..", "..", "calib"))
    polyCalib = ReconstructCalibration(data_folder)
    polyCalib.calibrate_all()
