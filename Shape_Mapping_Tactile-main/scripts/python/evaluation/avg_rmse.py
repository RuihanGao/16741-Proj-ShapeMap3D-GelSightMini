#!/usr/bin/python

# Sudharshan Suresh (suddhu@cmu.edu), Sep 2020
# loads the saved RMSE values from txt and plots the avg RMSE (trans and rot) for all data logs

## method 1: python ./scripts/py/avg_rmse.py 50
# assumes /home/suddhu/planar-push-slam/results/ to be the results directory

## method 2: python ./scripts/py/avg_rmse.py /home/suddhu/rpl/icra-2021/results/sep_sim_50/ 50
# references external results directory

import scipy.io
import pdb
import numpy as np
from numpy import array as npa
import matplotlib.pyplot as plt
import glob, os, sys
import json

fig = plt.figure()

plt.rc('pdf',fonttype = 42)
plt.rc('ps',fonttype = 42)
plt.rc('font', family='serif')

# https://www.geeksforgeeks.org/python-print-common-elements-two-lists/
def common_member(a, b):
    a_set = set(a)
    b_set = set(b)

    if (a_set & b_set):
        return (a_set & b_set)
    else:
        return None

##############################################

shapes = ["rect1", "hex", "ellip2"]

shape_dict = {
    "rect1" : "red",
    "hex"   : "blue",
    "ellip2": "orange"
}

if len(sys.argv) == 3:
    dbpath = sys.argv[1]
    num_logs = int(sys.argv[2])
elif len(sys.argv) == 2:
    num_logs = int(sys.argv[1])
    dbpath = "/home/suddhu/planar-push-slam/results/"

if not os.path.exists(dbpath + "plots"):
    os.makedirs(dbpath + "plots")

save_path_txt = '%splots/RMSE.txt' % dbpath
save_path_plot = '%splots/RMSE_plot.pdf' % dbpath

s = 0
trans_rmse = np.zeros((3, num_logs))
rot_rmse = np.zeros((3, num_logs))
for shape in shapes:
    print("shape: ",shape)

    for i in range(0,num_logs):
        if i % 5 == 0:
            print('log', i + 1)
        else:
            sys.stdout.write('.'); sys.stdout.flush()

        path = dbpath + shape + "/all_contact_shape=" + shape + "_rep=" + str(i + 1).zfill(4) + "/rmse.txt"

        try:
            rmse_vals = np.loadtxt(path, delimiter=" ", unpack=False)
            trans_rmse[s][i] = rmse_vals[0]
            rot_rmse[s][i] = rmse_vals[1]
        except:
            print("file ", path, " not found\n")
            continue
    s = s + 1

# pdb.set_trace()
trans_rmse_mean = 1000*trans_rmse.mean(axis=1)
trans_rmse_std = 1000*trans_rmse.std(axis=1)
rot_rmse_mean = rot_rmse.mean(axis=1)
rot_rmse_std = rot_rmse.std(axis=1)
np.savetxt(save_path_txt, (trans_rmse_mean, trans_rmse_std, rot_rmse_mean, rot_rmse_std), fmt="%.2f")
print('txt saved to ', save_path_txt)

# error bar plots
print("trans_rmse_mean: ", trans_rmse_mean)
print("trans_rmse_std: ", trans_rmse_std)
print("rot_rmse_mean: ", rot_rmse_mean)
print("rot_rmse_std: ", rot_rmse_std)

i = 0
k = 10
for shape in shapes:
    print("shape: ",shape)
    min_trans = np.argpartition(trans_rmse[i,:], k)[:k] + 1
    min_rot = np.argpartition(rot_rmse[i,:], k)[:k] + 1
    print(shape, " best fits: ", common_member(min_trans, min_rot))
    i = i + 1

t = np.array([0, 1, 2])

ax1 = fig.add_subplot(1, 2, 1)

ax1.set_ylabel('Trans RMSE (mm)')
ax1.errorbar(t[0], trans_rmse_mean[0], yerr=trans_rmse_std[0], ecolor=shape_dict["rect1"], markerfacecolor=shape_dict["rect1"], markeredgecolor='k', fmt='o', capsize=5)
ax1.errorbar(t[1], trans_rmse_mean[1], yerr=trans_rmse_std[1], ecolor=shape_dict["hex"], markerfacecolor=shape_dict["hex"], markeredgecolor='k',  fmt='o', capsize=5)
ax1.errorbar(t[2], trans_rmse_mean[2], yerr=trans_rmse_std[2], ecolor=shape_dict["ellip2"], markerfacecolor=shape_dict["ellip2"], markeredgecolor='k',  fmt='o', capsize=5)

plt.xticks(t, shapes)

ax2 = fig.add_subplot(1, 2, 2)

ax2.set_ylabel('Rot RMSE (rad)')  # we already handled the x-label with ax1
ax2.errorbar(t[0], rot_rmse_mean[0], yerr=rot_rmse_std[0], ecolor=shape_dict["rect1"], markerfacecolor=shape_dict["rect1"], markeredgecolor='k', fmt='o', capsize=5)
ax2.errorbar(t[1], rot_rmse_mean[1], yerr=rot_rmse_std[1], ecolor=shape_dict["hex"], markerfacecolor=shape_dict["hex"], markeredgecolor='k',  fmt='o', capsize=5)
ax2.errorbar(t[2], rot_rmse_mean[2], yerr=rot_rmse_std[2], ecolor=shape_dict["ellip2"], markerfacecolor=shape_dict["ellip2"], markeredgecolor='k',  fmt='o', capsize=5)

plt.xticks(t, shapes)

ax1.yaxis.grid(True)
ax2.yaxis.grid(True)

fig.suptitle('RMSE over ' + str(num_logs) + ' trials', fontsize=14)
fig.tight_layout()
fig.subplots_adjust(top=0.92)
plt.savefig(save_path_plot, transparent = True, bbox_inches = 'tight', pad_inches = 0)
print('saved to ', save_path_plot)
