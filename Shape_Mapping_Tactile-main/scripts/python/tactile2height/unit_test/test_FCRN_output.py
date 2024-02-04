# import matplotlib.pyplot as plt
import numpy as np
import os
from os import path as osp
import cv2
import scipy

result_path = osp.join('..', '..','..', '..','..','..','results','outputs','test')
pred_path = osp.join(result_path, 'pred')
gt_path = osp.join(result_path, 'gt')
pred_folders = sorted(os.listdir(pred_path))
gt_folders = sorted(os.listdir(gt_path))
num_result = len(pred_folders)//2
all_err = 0
for idx in range(num_result):
#     print(img_dir)
    # pred_img = cv2.imread(osp.join(pred_path,pred_folders[idx]))
    # gt_img = cv2.imread(osp.join(gt_path,gt_folders[idx]))
    pred_img = np.load(pred_path+'/Test_pred_depth_{:05d}.npy'.format(idx))
    gt_img = np.load(gt_path+'/Test_gt_depth_{:05d}.npy'.format(idx))
    # print(gt_img.shape)
    # print(pred_img.shape)
    # print(np.mean(np.abs(pred_img-gt_img)))
    mse_err = np.square(np.subtract(pred_img,gt_img)).mean()
    print(mse_err)
    all_err += mse_err
    # plt.figure(1)
    # plt.subplot(211)
    # plt.imshow(gt_img)
    #
    # plt.subplot(212)
    # plt.imshow(pred_img)
    # plt.show()
    # p = input("pause")
avg_err = all_err/num_result
print("average mse is " + str(avg_err))
