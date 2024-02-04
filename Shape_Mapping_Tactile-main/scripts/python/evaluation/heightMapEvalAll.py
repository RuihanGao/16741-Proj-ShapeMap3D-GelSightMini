import os
from os import path as osp
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D

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

def IoU(mask1, mask2):
    and_area = np.sum(mask1 & mask2)
    or_area = np.sum(mask1 | mask2)
    return and_area.astype(float)/or_area.astype(float)
# std of each object


model1 = 'lookup' # 'fcrn' # 'lookup' # 'mlp'
model2 = 'mlp'
model3 = 'fcrn'
num_samples = 60

gt_path = osp.join('/Users','ZilinSi','Desktop','shape_mapping','github', 'Shape_Mapping_Tactile', 'gelsight_data','textured_60sampled')
gen_path = osp.join('/Users','ZilinSi','Desktop','shape_mapping','github', 'Shape_Mapping_Tactile', 'generated_data','textured_60sampled')


gen_heightMap_path1 = osp.join(gen_path, model1)
gen_heightMap_path2 = osp.join(gen_path, model2)
gen_heightMap_path3 = osp.join(gen_path, model3)
gen_contact_mask_path1 = osp.join(gen_path, 'contact_masks_old')
gen_contact_mask_path2 = osp.join(gen_path, 'contact_masks_old')
gen_contact_mask_path3 = osp.join(gen_path, 'contact_masks')

objects_old = sorted(os.listdir(gen_heightMap_path1))
objects = []
for oi, obj in enumerate(objects_old):
    if obj == ".DS_Store":
        continue
    objects.append(obj)
print(objects)

all_height_error1 = np.zeros((len(objects),num_samples))
all_height_mask_error1 = np.zeros((len(objects),num_samples))
all_height_error2 = np.zeros((len(objects),num_samples))
all_height_mask_error2 = np.zeros((len(objects),num_samples))
all_height_error3 = np.zeros((len(objects),num_samples))
all_height_mask_error3 = np.zeros((len(objects),num_samples))
for oi, obj in enumerate(objects):
    if obj == ".DS_Store":
        continue
    print(obj)
    gt_heightMap_folder = osp.join(gt_path, obj, 'gt_height_map')
    gt_contact_mask_folder = osp.join(gt_path, obj, 'gt_contact_mask')

    gen_heightMap_folder1 = osp.join(gen_heightMap_path1, obj)
    gen_heightMap_folder2 = osp.join(gen_heightMap_path2, obj)
    gen_heightMap_folder3 = osp.join(gen_heightMap_path3, obj)
    gen_contact_mask_folder1 = osp.join(gen_contact_mask_path1, obj)
    gen_contact_mask_folder2 = osp.join(gen_contact_mask_path2, obj)
    gen_contact_mask_folder3 = osp.join(gen_contact_mask_path3, obj)

    obj_height_error1 = []
    obj_height_mask_error1 = []
    obj_height_error2 = []
    obj_height_mask_error2 = []
    obj_height_error3 = []
    obj_height_mask_error3 = []
    for i in range(num_samples):
        gt_height_map = np.load(osp.join(gt_heightMap_folder, str(i)+'.npy'))
        gt_contact_mask = np.load(osp.join(gt_contact_mask_folder, str(i)+'.npy'))

        gen_heightMap1 = np.load(osp.join(gen_heightMap_folder1, str(i)+'.npy'))
        gen_heightMap2 = np.load(osp.join(gen_heightMap_folder2, str(i)+'.npy'))
        gen_heightMap3 = np.load(osp.join(gen_heightMap_folder3, str(i)+'.npy'))
        gen_contact_mask1 = np.load(osp.join(gen_contact_mask_folder1, str(i)+'.npy'))
        gen_contact_mask2 = np.load(osp.join(gen_contact_mask_folder2, str(i)+'.npy'))
        gen_contact_mask3 = np.load(osp.join(gen_contact_mask_folder3, str(i)+'.npy'))

        dim = (640, 480)
        gen_heightMap1 = cv2.resize(gen_heightMap1, dim, interpolation = cv2.INTER_AREA)
        gen_heightMap2 = cv2.resize(gen_heightMap2, dim, interpolation = cv2.INTER_AREA)
        gen_heightMap3 = cv2.resize(gen_heightMap3, dim, interpolation = cv2.INTER_AREA)
        gen_contact_mask1 = cv2.resize(gen_contact_mask1.astype('uint8'), dim, interpolation = cv2.INTER_AREA)
        gen_contact_mask2 = cv2.resize(gen_contact_mask2.astype('uint8'), dim, interpolation = cv2.INTER_AREA)
        gen_contact_mask3 = cv2.resize(gen_contact_mask3.astype('uint8'), dim, interpolation = cv2.INTER_AREA)
        gen_heightMap1 = gen_heightMap1#[20:-20,20:-20]
        gen_heightMap2 = gen_heightMap2#[20:-20,20:-20]
        gen_heightMap3 = gen_heightMap3#[20:-20,20:-20]

        gen_contact_mask1 = gen_contact_mask1#[20:-20,20:-20]
        gen_contact_mask2 = gen_contact_mask2#[20:-20,20:-20]
        gen_contact_mask3 = gen_contact_mask3#[20:-20,20:-20]
        gt_height_map = gt_height_map#[20:-20,20:-20]
        gt_contact_mask = gt_contact_mask#[20:-20,20:-20]

        gt_height_map *= 0.0295
        gen_heightMap1 *= 0.0295
        gen_heightMap2 *= 0.0295
        gen_heightMap3 *= 0.0295
        mse_height1 = RMSE(gt_height_map, gen_heightMap1)
        mse_height_mask1 = IoU(gt_contact_mask, gen_contact_mask1)
        # mse_height_mask1 = RMSE(gt_height_map*gt_contact_mask, gen_heightMap1*gen_contact_mask1)
        mse_height2 = RMSE(gt_height_map, gen_heightMap2)
        mse_height_mask2 = IoU(gt_contact_mask, gen_contact_mask2)
        # mse_height_mask2 = RMSE(gt_height_map*gt_contact_mask, gen_heightMap2*gen_contact_mask2)
        mse_height3 = RMSE(gt_height_map, gen_heightMap3)
        mse_height_mask3 = IoU(gt_contact_mask, gen_contact_mask3)
        # mse_height_mask3 = RMSE(gt_height_map*gt_contact_mask, gen_heightMap3*gen_contact_mask3)
        # print("mse of masked height is " + str(mse_height_mask))

        obj_height_error1.append(mse_height1)
        obj_height_mask_error1.append(mse_height_mask1)
        obj_height_error2.append(mse_height2)
        obj_height_mask_error2.append(mse_height_mask2)
        obj_height_error3.append(mse_height3)
        obj_height_mask_error3.append(mse_height_mask3)

        all_height_error1[oi,i] = mse_height1
        all_height_mask_error1[oi,i] = mse_height_mask1
        all_height_error2[oi,i] = mse_height2
        all_height_mask_error2[oi,i] = mse_height_mask2
        all_height_error3[oi,i] = mse_height3
        all_height_mask_error3[oi,i] = mse_height_mask3
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
    avg_height_error1 = np.mean(obj_height_error1)
    avg_height_mask_error1 = np.mean(obj_height_mask_error1)
    std_height_error1 = std(obj_height_error1)
    std_height_mask_error1 = std(obj_height_mask_error1)

    avg_height_error2 = np.mean(obj_height_error2)
    avg_height_mask_error2 = np.mean(obj_height_mask_error2)
    std_height_error2 = std(obj_height_error2)
    std_height_mask_error2 = std(obj_height_mask_error2)

    avg_height_error3 = np.mean(obj_height_error3)
    avg_height_mask_error3 = np.mean(obj_height_mask_error3)
    std_height_error3 = std(obj_height_error3)
    std_height_mask_error3 = std(obj_height_mask_error3)
    # print(avg_height_error)
    # print(avg_height_mask_error)
    # print(std_height_error)
    # print(std_height_mask_error)

# pdb.set_trace()
height_rmse_mean1 = all_height_error1.mean(axis=1)
height_rmse_std1 = all_height_error1.std(axis=1)
height_mask_rmse_mean1 = all_height_mask_error1.mean(axis=1)
height_mask_rmse_std1 = all_height_mask_error1.std(axis=1)

height_rmse_mean2 = all_height_error2.mean(axis=1)
height_rmse_std2 = all_height_error2.std(axis=1)
height_mask_rmse_mean2 = all_height_mask_error2.mean(axis=1)
height_mask_rmse_std2 = all_height_mask_error2.std(axis=1)

height_rmse_mean3 = all_height_error3.mean(axis=1)
height_rmse_std3 = all_height_error3.std(axis=1)
height_mask_rmse_mean3 = all_height_mask_error3.mean(axis=1)
height_mask_rmse_std3 = all_height_mask_error3.std(axis=1)

height_mean_mean1 = np.mean(height_rmse_mean1)
height_mask_mean_mean1 = np.mean(height_mask_rmse_mean1)
height_mean_mean2 = np.mean(height_rmse_mean2)
height_mask_mean_mean2 = np.mean(height_mask_rmse_mean2)
height_mean_mean3 = np.mean(height_rmse_mean3)
height_mask_mean_mean3 = np.mean(height_mask_rmse_mean3)
print(height_mean_mean1)
print(height_mask_mean_mean1)

print(height_mean_mean2)
print(height_mask_mean_mean2)

print(height_mean_mean3)
print(height_mask_mean_mean3)

save_path_txt = osp.join('/Users','ZilinSi','Desktop','shape_mapping','github', 'Shape_Mapping_Tactile', 'results', 'all_height_mask_rmse.txt')
save_path_plot = osp.join('/Users','ZilinSi','Desktop','shape_mapping','github', 'Shape_Mapping_Tactile', 'results','all_height_mask_rmse.pdf')
# np.savetxt(save_path_txt, (height_rmse_mean, height_rmse_std, height_mask_rmse_mean, height_mask_rmse_std), fmt="%.2f")
# print('txt saved to ', save_path_txt)

# error bar plots
fig = plt.figure()

plt.rc('pdf',fonttype = 42)
plt.rc('ps',fonttype = 42)
plt.rc('font', family='serif')

test_object = ["002_master_chef_can", "004_sugar_box", "005_tomato_soup_can", "006_mustard_bottle", "010_potted_meat_can", "021_bleach_cleanser", "036_wood_block"]

shape_dict = {
    "test" : "black",
    "others"   : "cornflowerblue"
}

shape_dict = {
    "lookup" : "#5C3658",
    "mlp"   : "#CF515C",
    "fcrn"  : "#FFA600",
    "mask_fcrn": "#1B3E8F",
    "mask": "#8F92C0"
}

t = np.arange(len(objects))
figw = 12
figh = 5

ax1 = fig.add_subplot(2, 1, 1)
ax1.figure.set_size_inches(figw, figh)
ax1.set_ylim([0.0, 1.5])
ax1.axhline(y=height_mean_mean1, color=shape_dict["lookup"], alpha=0.5, linestyle='--', linewidth=0.8)
ax1.axhline(y=height_mean_mean2, color=shape_dict["mlp"], alpha=0.5, linestyle='--', linewidth=0.8)
ax1.axhline(y=height_mean_mean3, color=shape_dict["fcrn"], alpha=0.5, linestyle='--', linewidth=0.8)

# ax1.set_ylabel('HeightMap RMSE (mm)')
for i, obj in enumerate(objects):
    if obj in test_object:
        trans1 = Affine2D().translate(-0.05, 0.0) + ax1.transData
        trans2 = Affine2D().translate(+0.0, 0.0) + ax1.transData
        trans3 = Affine2D().translate(+0.05, 0.0) + ax1.transData
        ax1.errorbar(t[i], height_rmse_mean1[i], yerr=height_rmse_std1[i], ecolor=shape_dict["lookup"], markerfacecolor=shape_dict["lookup"], markeredgecolor='k', fmt='o', capsize=5, transform=trans1)
        ax1.errorbar(t[i], height_rmse_mean2[i], yerr=height_rmse_std2[i], ecolor=shape_dict["mlp"], markerfacecolor=shape_dict["mlp"], markeredgecolor='k', fmt='o', capsize=5, transform=trans2)
        ax1.errorbar(t[i], height_rmse_mean3[i], yerr=height_rmse_std3[i], ecolor=shape_dict["fcrn"], markerfacecolor=shape_dict["fcrn"], markeredgecolor='k', fmt='o', capsize=5, transform=trans3)
    else:
        trans1 = Affine2D().translate(-0.05, 0.0) + ax1.transData
        trans2 = Affine2D().translate(+0.0, 0.0) + ax1.transData
        trans3 = Affine2D().translate(+0.05, 0.0) + ax1.transData
        ax1.errorbar(t[i], height_rmse_mean1[i], yerr=height_rmse_std1[i], ecolor=shape_dict["lookup"], markerfacecolor=shape_dict["lookup"], markeredgecolor='k', fmt='o', capsize=5, transform=trans1)
        ax1.errorbar(t[i], height_rmse_mean2[i], yerr=height_rmse_std2[i], ecolor=shape_dict["mlp"], markerfacecolor=shape_dict["mlp"], markeredgecolor='k', fmt='o', capsize=5, transform=trans2)
        ax1.errorbar(t[i], height_rmse_mean3[i], yerr=height_rmse_std3[i], ecolor=shape_dict["fcrn"], markerfacecolor=shape_dict["fcrn"], markeredgecolor='k', fmt='o', capsize=5, transform=trans3)
        # ax1.errorbar(t[i], height_rmse_mean[i], yerr=height_rmse_std[i], ecolor=shape_dict["others"], markerfacecolor=shape_dict["others"], markeredgecolor='k', fmt='o', capsize=5)

    # ax1.errorbar(t[i], height_rmse_mean[i], yerr=height_rmse_std[i], ecolor=shape_dict["rect1"], markerfacecolor=shape_dict["rect1"], markeredgecolor='k', fmt='o', capsize=5)

# plt.xticks(t, objects, rotation=45, fontsize=8)
plt.xticks(t, t, fontsize=8)

ax2 = fig.add_subplot(2, 1, 2)
ax2.figure.set_size_inches(figw, figh)
ax2.set_ylim([0.0, 1.0])
ax2.axhline(y=height_mask_mean_mean1, color=shape_dict["mask"], alpha=0.5, linestyle='--', linewidth=0.8)
# ax2.axhline(y=height_mask_mean_mean2, color=shape_dict["mlp"], alpha=0.5, linestyle='--')
ax2.axhline(y=height_mask_mean_mean3, color=shape_dict["mask_fcrn"], alpha=0.5, linestyle='--', linewidth=0.8)

# ax2.set_ylabel('Masked HeightMap RMSE (mm)')  # we already handled the x-label with ax1
for i, obj in enumerate(objects):
    if obj in test_object:
        trans1 = Affine2D().translate(-0.025, 0.0) + ax2.transData
        trans3 = Affine2D().translate(+0.025, 0.0) + ax2.transData
        ax2.errorbar(t[i], height_mask_rmse_mean1[i], yerr=height_mask_rmse_std1[i], ecolor=shape_dict["mask"], markerfacecolor=shape_dict["mask"], markeredgecolor='k', fmt='o', capsize=5, transform=trans1)
        ax2.errorbar(t[i], height_mask_rmse_mean3[i], yerr=height_mask_rmse_std3[i], ecolor=shape_dict["mask_fcrn"], markerfacecolor=shape_dict["mask_fcrn"], markeredgecolor='k', fmt='o', capsize=5, transform=trans3)
    else:
        trans1 = Affine2D().translate(-0.025, 0.0) + ax2.transData
        trans3 = Affine2D().translate(+0.025, 0.0) + ax2.transData
        ax2.errorbar(t[i], height_mask_rmse_mean1[i], yerr=height_mask_rmse_std1[i], ecolor=shape_dict["mask"], markerfacecolor=shape_dict["mask"], markeredgecolor='k', fmt='o', capsize=5, transform=trans1)
        ax2.errorbar(t[i], height_mask_rmse_mean3[i], yerr=height_mask_rmse_std3[i], ecolor=shape_dict["mask_fcrn"], markerfacecolor=shape_dict["mask_fcrn"], markeredgecolor='k', fmt='o', capsize=5, transform=trans3)

# plt.xticks(t, objects, rotation=45, fontsize=8)
plt.xticks(t, t, fontsize=8)

ax1.yaxis.grid(True)
ax2.yaxis.grid(True)

# fig.suptitle('RMSE over ' + str(num_samples) + ' trials', fontsize=14)
fig.tight_layout()
fig.subplots_adjust(top=0.92)
plt.savefig(save_path_plot, transparent = True, bbox_inches = 'tight', pad_inches = 0)
print('saved to ', save_path_plot)
