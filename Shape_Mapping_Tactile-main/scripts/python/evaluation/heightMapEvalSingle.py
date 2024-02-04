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

def std(value_list):
    return np.std(value_list)

# std of each object


model = 'fcrn' # 'fcrn' # 'lookup' # 'mlp'
num_samples = 60

gt_path = osp.join('/Users','ZilinSi','Desktop','shape_mapping','github', 'Shape_Mapping_Tactile', 'gelsight_data','textured_60sampled')
gen_path = osp.join('/Users','ZilinSi','Desktop','shape_mapping','github', 'Shape_Mapping_Tactile', 'generated_data','textured_60sampled')


gen_heightMap_path = osp.join(gen_path, model)
if model == 'mlp':
    gen_contact_mask_path = osp.join(gen_path, 'contact_masks_old')
elif model == 'lookup':
    gen_contact_mask_path = osp.join(gen_path, 'contact_masks_old')
else:
    gen_contact_mask_path = osp.join(gen_path, 'contact_masks') # fcrn

objects_old = sorted(os.listdir(gen_heightMap_path))
objects = []
for oi, obj in enumerate(objects_old):
    if obj == ".DS_Store":
        continue
    objects.append(obj)
print(objects)

all_height_error = np.zeros((len(objects),num_samples))
all_height_mask_error = np.zeros((len(objects),num_samples))
for oi, obj in enumerate(objects):
    if obj == ".DS_Store":
        continue
    print(obj)
    gt_heightMap_folder = osp.join(gt_path, obj, 'gt_height_map')
    gt_contact_mask_folder = osp.join(gt_path, obj, 'gt_contact_mask')

    gen_heightMap_folder = osp.join(gen_heightMap_path, obj)
    gen_contact_mask_folder = osp.join(gen_contact_mask_path, obj)

    obj_height_error = []
    obj_height_mask_error = []
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
        gen_contact_mask = cv2.resize(gen_contact_mask.astype('uint8'), dim, interpolation = cv2.INTER_AREA)
        gen_heightMap = gen_heightMap[20:-20,20:-20]
        gen_contact_mask = gen_contact_mask[20:-20,20:-20]
        gt_height_map = gt_height_map[20:-20,20:-20]
        gt_contact_mask = gt_contact_mask[20:-20,20:-20]

        gt_height_map *= 0.0295
        gen_heightMap *= 0.0295
        mse_height = RMSE(gt_height_map, gen_heightMap)
        mse_height_mask = RMSE(gt_height_map*gt_contact_mask, gen_heightMap*gen_contact_mask)
        # print("mse of masked height is " + str(mse_height_mask))

        obj_height_error.append(mse_height)
        obj_height_mask_error.append(mse_height_mask)
        all_height_error[oi,i] = mse_height
        all_height_mask_error[oi,i] = mse_height_mask
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
    avg_height_error = np.mean(obj_height_error)
    avg_height_mask_error = np.mean(obj_height_mask_error)
    std_height_error = std(obj_height_error)
    std_height_mask_error = std(obj_height_mask_error)
    print(avg_height_error)
    print(avg_height_mask_error)
    print(std_height_error)
    print(std_height_mask_error)
    # all_height_error.append(avg_height_error)
    # all_height_mask_error.append(avg_height_mask_error)

# pdb.set_trace()
height_rmse_mean = all_height_error.mean(axis=1)
height_rmse_std = all_height_error.std(axis=1)
height_mask_rmse_mean = all_height_mask_error.mean(axis=1)
height_mask_rmse_std = all_height_mask_error.std(axis=1)
height_mean_mean = np.mean(height_rmse_mean)
height_mask_mean_mean = np.mean(height_mask_rmse_mean)

save_path_txt = osp.join('/Users','ZilinSi','Desktop','shape_mapping','github', 'Shape_Mapping_Tactile', 'results', model+'_height.txt')
save_path_plot = osp.join('/Users','ZilinSi','Desktop','shape_mapping','github', 'Shape_Mapping_Tactile', 'results',model+'_height.pdf')
# np.savetxt(save_path_txt, (height_rmse_mean, height_rmse_std, height_mask_rmse_mean, height_mask_rmse_std), fmt="%.2f")
# print('txt saved to ', save_path_txt)

# error bar plots
print("height_rmse_mean: ", height_rmse_mean)
print("height_rmse_std: ", height_rmse_std)
print("height_mask_rmse_mean: ", height_mask_rmse_mean)
print("height_mask_rmse_std: ", height_mask_rmse_std)

fig = plt.figure()

plt.rc('pdf',fonttype = 42)
plt.rc('ps',fonttype = 42)
plt.rc('font', family='serif')

test_object = ["002_master_chef_can", "004_sugar_box", "005_tomato_soup_can", "006_mustard_bottle", "010_potted_meat_can", "021_bleach_cleanser", "036_wood_block"]

shape_dict = {
    "test" : "black",
    "others"   : "cornflowerblue"
}

t = np.arange(len(objects))
figw = 10
figh = 5

ax1 = fig.add_subplot(2, 1, 1)
ax1.figure.set_size_inches(figw, figh)
ax1.set_ylim([0.0, 1.0])
ax1.axhline(y=height_mean_mean, color='gold', linestyle='--')

# ax1.set_ylabel('HeightMap RMSE (mm)')
for i, obj in enumerate(objects):
    if obj in test_object:
        ax1.errorbar(t[i], height_rmse_mean[i], yerr=height_rmse_std[i], ecolor=shape_dict["test"], markerfacecolor=shape_dict["test"], markeredgecolor='k', fmt='o', capsize=5)
    else:
        ax1.errorbar(t[i], height_rmse_mean[i], yerr=height_rmse_std[i], ecolor=shape_dict["others"], markerfacecolor=shape_dict["others"], markeredgecolor='k', fmt='o', capsize=5)

    # ax1.errorbar(t[i], height_rmse_mean[i], yerr=height_rmse_std[i], ecolor=shape_dict["rect1"], markerfacecolor=shape_dict["rect1"], markeredgecolor='k', fmt='o', capsize=5)

# plt.xticks(t, objects, rotation=45, fontsize=8)
plt.xticks(t, t, fontsize=8)

ax2 = fig.add_subplot(2, 1, 2)
ax2.figure.set_size_inches(figw, figh)
ax2.set_ylim([0.0, 1.0])
ax2.axhline(y=height_mask_mean_mean, color='gold', linestyle='--')
# ax2.set_ylabel('Masked HeightMap RMSE (mm)')  # we already handled the x-label with ax1
for i, obj in enumerate(objects):
    if obj in test_object:
        ax2.errorbar(t[i], height_mask_rmse_mean[i], yerr=height_mask_rmse_std[i], ecolor=shape_dict["test"], markerfacecolor=shape_dict["test"], markeredgecolor='k', fmt='o', capsize=5)
    else:
        ax2.errorbar(t[i], height_mask_rmse_mean[i], yerr=height_mask_rmse_std[i], ecolor=shape_dict["others"], markerfacecolor=shape_dict["others"], markeredgecolor='k', fmt='o', capsize=5)

# plt.xticks(t, objects, rotation=45, fontsize=8)
plt.xticks(t, t, fontsize=8)

ax1.yaxis.grid(True)
ax2.yaxis.grid(True)

# fig.suptitle('RMSE over ' + str(num_samples) + ' trials', fontsize=14)
fig.tight_layout()
fig.subplots_adjust(top=0.92)
plt.savefig(save_path_plot, transparent = True, bbox_inches = 'tight', pad_inches = 0)
print('saved to ', save_path_plot)


# plot all errors with bar
# fig = plt.figure(figsize = (10, 5))
# plt.bar(objects, all_height_error, color ='maroon',
#         width = 0.8)
#
# plt.xlabel("object")
# plt.ylabel("pixel-wised height map error (mm)")
# plt.xticks(rotation=30 )
# plt.show()
#
# fig = plt.figure(figsize = (10, 5))
# plt.bar(objects, all_height_mask_error, color ='maroon',
#         width = 0.8)
#
# plt.xlabel("object")
# plt.ylabel("pixel-wised masked height map error (mm)")
# plt.xticks(rotation=30 )
# plt.show()
