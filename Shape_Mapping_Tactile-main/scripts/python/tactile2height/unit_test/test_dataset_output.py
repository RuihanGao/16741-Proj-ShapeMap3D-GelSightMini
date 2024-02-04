import matplotlib.pyplot as plt
import numpy as np
import os
from os import path as osp
import cv2
import scipy

data_folder = '../../../../generated_data/real/fcrn/'
tactile_folder = '../../../../gelsight_data/real/'

object = '005_tomato_soup_can'
if object is None:
    object_folders = sorted(os.listdir(save_path))
else:
    object_folders = [object]

for obj in object_folders:
    if obj == ".DS_Store":
        continue
    data_folder = data_folder + obj + '/'
    tactile_folder = tactile_folder + obj + '/'
    heightMaps = sorted(os.listdir(data_folder), key=lambda y: int(y.split(".")[0]))
    tactiles = sorted(os.listdir(tactile_folder), key=lambda y: int(y.split("_")[1]))
    for i in range(2):
        height = np.load(data_folder + heightMaps[i])
        tactile = cv2.imread(tactile_folder + tactiles[i])
        tactile = cv2.cvtColor(tactile, cv2.COLOR_BGR2RGB)
        plt.figure(1)
        plt.subplot(211)
        plt.imshow(height)

        plt.subplot(212)
        plt.imshow(tactile/255.0)
        plt.show()
        p = input("pause")
