import matplotlib.pyplot as plt
import numpy as np
import os
from os import path as osp
import cv2
import scipy


def simulate(data_folder, mask_folder, img_folder, object=None):
    if object is None:
        # generate height maps for all objects
        object_folders = sorted(os.listdir(data_folder))
    else:
        object_folders = [object]

    for obj in object_folders:
        if obj == ".DS_Store":
            continue
        print(obj)
        cur_folder = data_folder + obj + '/'
        tactile_folder = img_folder + obj + '/'
        mask_folder = mask_folder + obj + '/'
        heightMaps = sorted(os.listdir(cur_folder), key=lambda y: int(y.split(".")[0]))
        tactiles = sorted(os.listdir(tactile_folder), key=lambda y: int(y.split("_")[1]))
        masks = sorted(os.listdir(mask_folder), key=lambda y: int(y.split(".")[0]))

        for i, heightMap in enumerate(heightMaps):
            if i % 2 == 0:
                continue
            print(heightMap)
            # img = cv2.imread(data_folder + str(idx) + ‘.jpg’)
            # img = img.astype(int)

            img_path = cur_folder + heightMap
            last_tactile = tactile_folder + tactiles[i-1]
            tactile = tactile_folder + tactiles[i]
            mask_path = mask_folder + masks[i]

            height_map = np.load(img_path)
            last_tactile_img = cv2.imread(last_tactile)
            tactile_img = cv2.imread(tactile)
            mask = np.load(mask_path)

            diff_img = np.abs(np.sum(tactile_img.astype(float) - last_tactile_img.astype(float),axis=2))
            # plt.imshow(height_map)
            # plt.show()

            plt.figure(1)
            plt.subplot(221)
            fig = plt.imshow(height_map)

            plt.subplot(222)
            fig = plt.imshow(tactile_img/255.0)

            plt.subplot(223)
            fig = plt.imshow(diff_img/255.0)

            plt.subplot(224)
            fig = plt.imshow(mask)

            plt.show()
            # print(height_map.shape)
            # print(np.max(height_map))
            # print(np.min(height_map))

            # fig = plt.figure(1)
            # ax1 = fig.add_subplot(221)
            # ax1.title.set_text('Image')
            # plt.imshow(img)

            # ax2 = fig.add_subplot(222)
            # ax2.title.set_text('FCRN heightmap')
            # plt.imshow(height_map)

            # plt.show()



if __name__ == "__main__":

    data_folder = '../../../../generated_data/real/fcrn/'
    mask_folder = '../../../../generated_data/real/contact_mask_fcrn/'
    # mask_folder = '../../../../generated_data/real/fcrn/'
    tactile_folder = '../../../../gelsight_data/real/'
    obj = '036_wood_block'
    simulate(data_folder, mask_folder, tactile_folder,object=obj)
