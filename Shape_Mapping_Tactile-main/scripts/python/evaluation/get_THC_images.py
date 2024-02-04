import matplotlib.pyplot as plt
import numpy as np
import os
from os import path as osp
import cv2
import scipy

save_path_plot = osp.join('/Users','ZilinSi','Desktop','shape_mapping','github', 'Shape_Mapping_Tactile', 'results','images')

def simulate(data_folder, mask_folder, img_folder, sim=True, object=None):
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
        if sim:
            tactiles = sorted(os.listdir(tactile_folder), key=lambda y: int(y.split(".")[0]))
            sim_type = "sim"
        else:
            tactiles = sorted(os.listdir(tactile_folder), key=lambda y: int(y.split("_")[1]))
            sim_type = "real"
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
            mask = 1*(mask > 0)
            if i == 1:
                tactile_img = cv2.cvtColor(tactile_img, cv2.COLOR_BGR2RGB)
                tactile_img = cv2.resize(tactile_img, (320,240), interpolation = cv2.INTER_AREA)
                plt.imshow(tactile_img/255.0)
                plt.imsave(osp.join(save_path_plot, sim_type+"_"+obj+"_tactile.jpg"), tactile_img)

                plt.imshow(height_map)
                plt.imsave(osp.join(save_path_plot,sim_type+"_"+obj+"height.jpg"), height_map, cmap = "viridis")

                blur=((5,5),1)
                erode_=(5,5)
                dilate_=(3, 3)
                mask = cv2.GaussianBlur(mask.astype('uint8'), blur[0], blur[1])
                mask = cv2.GaussianBlur(mask.astype('uint8'), blur[0], blur[1])
                # mask = cv2.dilate(cv2.erode(cv2.GaussianBlur(mask.astype('uint8'), blur[0], blur[1]), np.ones(erode_)), np.ones(dilate_))*255
                plt.imshow(mask)
                plt.imsave(osp.join(save_path_plot,sim_type+"_"+obj+"mask.jpg"), ~mask, cmap = "binary")
                break


            # plt.figure(1)
            # plt.subplot(221)
            # fig = plt.imshow(height_map)
            #
            # plt.subplot(222)
            # fig = plt.imshow(tactile_img/255.0)
            #
            # plt.subplot(223)
            # fig = plt.imshow(diff_img/255.0)
            #
            # plt.subplot(224)
            # fig = plt.imshow(mask)
            #
            # plt.show()



if __name__ == "__main__":

    data_folder = '../../../generated_data/real/fcrn/'
    mask_folder = '../../../generated_data/real/contact_mask_fcrn/'
    tactile_folder = '../../../gelsight_data/real/'
    sim = False
    # data_folder = '../../../generated_data/textured_60sampled/fcrn/'
    # mask_folder = '../../../generated_data/textured_60sampled/contact_masks/'
    # tactile_folder = '../../../gelsight_data/textured_60sampled/'
    # sim = True
    obj = '002_master_chef_can'
    simulate(data_folder, mask_folder, tactile_folder, sim=sim, object=obj)
