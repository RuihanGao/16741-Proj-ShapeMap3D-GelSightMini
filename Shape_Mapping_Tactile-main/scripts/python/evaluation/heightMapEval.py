import os
from os import path as osp
import numpy as np
import cv2
import matplotlib.pyplot as plt

def MSE(img1, img2):
    squared_diff = (img1 -img2) ** 2
    summed = np.sum(squared_diff)
    num_pix = img1.shape[0] * img1.shape[1] #img1 and 2 should have same shape
    err = summed / num_pix
    return err


def RMSE(img1, img2):
    squared_diff = (img1 -img2) ** 2
    err = np.sqrt(np.mean(squared_diff))
    return err

# std of each object


model = 'fcrn' # 'fcrn' # 'lookup' # 'mlp'
num_samples = 60

gt_path = osp.join('/Users','ZilinSi','Desktop','shape_mapping','github', 'Shape_Mapping_Tactile', 'gelsight_data','textured_60sampled')
gen_path = osp.join('/Users','ZilinSi','Desktop','shape_mapping','github', 'Shape_Mapping_Tactile', 'generated_data','textured_60sampled')


gen_heightMap_path = osp.join(gen_path, model)
gen_contact_mask_path = osp.join(gen_path, 'contact_masks')

objects = sorted(os.listdir(gen_heightMap_path))
objects = ['036_wood_block']

all_height_error = []
all_height_mask_error = []
for obj in objects:
    if obj == ".DS_Store":
        continue
    print(obj)
    gt_heightMap_folder = osp.join(gt_path, obj, 'gt_height_map')
    gt_contact_mask_folder = osp.join(gt_path, obj, 'gt_contact_mask')

    gen_heightMap_folder = osp.join(gen_heightMap_path, obj)
    gen_contact_mask_folder = osp.join(gen_contact_mask_path, obj)

    obj_height_error = 0.0
    obj_height_mask_error = 0.0
    for i in range(num_samples):
        gt_height_map = np.load(osp.join(gt_heightMap_folder, str(i)+'.npy'))
        gt_contact_mask = np.load(osp.join(gt_contact_mask_folder, str(i)+'.npy'))

        gen_heightMap = np.load(osp.join(gen_heightMap_folder, str(i)+'.npy'))
        gen_contact_mask = np.load(osp.join(gen_contact_mask_folder, str(i)+'.npy'))

        # dim = (320, 240)
        # gen_heightMap = cv2.resize(gen_heightMap, dim, interpolation = cv2.INTER_AREA)
        # gt_height_map = cv2.resize(gt_height_map, dim, interpolation = cv2.INTER_AREA)
        # gt_contact_mask = cv2.resize(gt_contact_mask.astype('uint8'), dim, interpolation = cv2.INTER_AREA)
        # gen_contact_mask = cv2.resize(gen_contact_mask.astype('uint8'), dim, interpolation = cv2.INTER_AREA)

        dim = (640, 480)
        gen_heightMap = cv2.resize(gen_heightMap, dim, interpolation = cv2.INTER_AREA)
        gen_heightMap = gen_heightMap[20:-20,20:-20]
        gt_height_map = gt_height_map[20:-20,20:-20]
        gt_contact_mask = gt_contact_mask[20:-20,20:-20]

        gt_height_map *= 0.0295
        gen_heightMap *= 0.0295
        mse_height = MSE(gt_height_map, gen_heightMap)
        mse_height_mask = MSE(gt_height_map*gt_contact_mask, gen_heightMap*gen_contact_mask)
        # print("mse of masked height is " + str(mse_height_mask))

        obj_height_error += mse_height
        obj_height_mask_error += mse_height_mask

        # plt.figure(1)
        # plt.subplot(221)
        # plt.imshow(gt_height_map*gt_contact_mask)
        #
        # plt.subplot(222)
        # plt.imshow(gen_heightMap*gen_contact_mask)
        #
        # plt.subplot(223)
        # plt.imshow(gt_height_map)
        #
        # plt.subplot(224)
        # plt.imshow(gen_heightMap)
        # plt.show()
        # p = input("pause")
    obj_height_error /= num_samples
    obj_height_mask_error /= num_samples
    print(obj_height_error)
    print(obj_height_mask_error)
    all_height_error.append(obj_height_error)
    all_height_mask_error.append(obj_height_mask_error)

# plot all errors
fig = plt.figure(figsize = (10, 5))
plt.bar(objects, all_height_error, color ='maroon',
        width = 0.8)

plt.xlabel("object")
plt.ylabel("pixel-wised height map error (mm)")
plt.xticks(rotation=30 )
plt.show()

fig = plt.figure(figsize = (10, 5))
plt.bar(objects, all_height_mask_error, color ='maroon',
        width = 0.8)

plt.xlabel("object")
plt.ylabel("pixel-wised masked height map error (mm)")
plt.xticks(rotation=30 )
plt.show()
