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

def processInitialFrame(f0):
    # gaussian filtering with square kernel with
    # filterSize : kscale*2+1
    # sigma      : kscale
    kscale = pr.kscale

    img_d = f0.astype('float')
    convEachDim = lambda in_img :  gaussian_filter(in_img, kscale)

    f0 = f0.copy()
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

calib_folder = osp.join('..', '..','..','..','calib') # tactile img config file
rawData = osp.join(calib_folder, "dataPack.npz")
data_file = np.load(rawData,allow_pickle=True)
f0 = data_file['f0']
bg_proc = processInitialFrame(f0)
cv2.imwrite(osp.join(calib_folder, "bg.jpg"),bg_proc)