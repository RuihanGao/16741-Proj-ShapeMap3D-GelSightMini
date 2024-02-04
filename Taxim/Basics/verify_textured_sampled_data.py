# %% [markdown]
# # Read the textured_50sampled data and verify the reproducibility with the original implementation
# (.mat for original implementation and .npz for Ruihan's reimplementation in Python)

# %%
import scipy.io as sio
import numpy as np
import os.path as osp
import sys
import os

# # %% [markdown]
# # ## Case 1: existing data. compare org_data and my_data

# # %%
# obj_name = "006_mustard_bottle"
# data_dir  = f"/home/ruihan/Documents/16741_proj/Shape_Mapping_Tactile-main/models/{obj_name}"
# org_data = sio.loadmat(osp.join(data_dir, 'textured_50sampled.mat'))
# my_data = sio.loadmat(osp.join(data_dir, 'textured_60sampled_python.mat'))

# # %%
# # compare the keys
# print(f"org_data: {org_data.keys()}")
# print(f"my_data: {my_data.keys()}")
# print(f"compare number of samples {org_data['numSamples']}  vs {my_data['numSamples']}")
# org_sample_points = org_data['samplePoints']
# my_sample_points = my_data['samplePoints']
# print(f"compare sampled points {org_sample_points.shape} vs {my_sample_points.shape}")
# org_sample_normals = org_data['sampleNormals']
# my_sample_normals = my_data['sampleNormals']
# print(f"compare sampled normals {org_sample_normals.shape} vs {my_sample_normals.shape}")

# # %%
# # Plot the original sampled points and new sampled points in the same 3D plot
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from sampleModelPoints import set_axes_equal
# fig = plt.figure()
# ax = Axes3D(fig)
# ax.scatter(org_sample_points[:,0], org_sample_points[:,1], org_sample_points[:,2], c='r', marker='o')
# ax.scatter(my_sample_points[:,0], my_sample_points[:,1], my_sample_points[:,2], c='b', marker='^')
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
# set_axes_equal(ax)
# # Add legend
# red_patch = plt.Line2D([0], [0], linestyle="none", c='r', marker='o')
# blue_patch = plt.Line2D([0], [0], linestyle="none", c='b', marker='^')
# plt.legend([red_patch, blue_patch], ['original sampled points', 'new sampled points'])
# plt.title('Sampled points and normals')
# plt.savefig(osp.join(data_dir, 'compare_sampled_points.png'), dpi=600)
# plt.show()


# # %%
# # Plot the original sampled points and new sampled points in the same 3D plot
# # Add the sampled normals upon the 3D points
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from sampleModelPoints import set_axes_equal
# fig = plt.figure()
# ax = Axes3D(fig)
# ax.scatter(org_sample_points[:,0], org_sample_points[:,1], org_sample_points[:,2], c='r', marker='o')
# # add sampled normals
# ax.quiver(org_sample_points[:,0], org_sample_points[:,1], org_sample_points[:,2], org_sample_normals[:,0], org_sample_normals[:,1], org_sample_normals[:,2], length=0.01, color='r')
# ax.scatter(my_sample_points[:,0], my_sample_points[:,1], my_sample_points[:,2], c='b', marker='^')
# # add sampled normals
# ax.quiver(my_sample_points[:,0], my_sample_points[:,1], my_sample_points[:,2], my_sample_normals[:,0], my_sample_normals[:,1], my_sample_normals[:,2], length=0.01, color='b')
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
# set_axes_equal(ax)
# # Add legend
# red_patch = plt.Line2D([0], [0], linestyle="none", c='r', marker='o')
# blue_patch = plt.Line2D([0], [0], linestyle="none", c='b', marker='^')
# # add legend for the normals
# plt.legend([red_patch, blue_patch], ['original sampled points', 'new sampled points'])
# plt.title('Sampled points and normals')
# plt.savefig(osp.join(data_dir, 'compare_sampled_points_normals.png'), dpi=600)
# plt.show()


# %% [markdown]
# ## Case 2. New data. Visualize my_data

# %%
obj_name = "calib_ball"
data_dir  = f"/home/ruihan/Documents/16741_proj/Shape_Mapping_Tactile-main/models/{obj_name}"
my_data = sio.loadmat(osp.join(data_dir, 'textured_60sampled_python.mat'))

# %%
print(f"my_data: {my_data.keys()}")
my_sample_points = my_data['samplePoints']
my_sample_normals = my_data['sampleNormals']
print(f"compare my_sample_points {my_sample_points.shape}, my_sample_normals {my_sample_normals.shape}")

# %%
# Plot the sampled points and the sampled normals upon the 3D points
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sampleModelPoints import set_axes_equal
fig = plt.figure()
ax = Axes3D(fig)
# add sampled points
ax.scatter(my_sample_points[:,0], my_sample_points[:,1], my_sample_points[:,2], c='b', marker='o')
# add sampled normals
ax.quiver(my_sample_points[:,0], my_sample_points[:,1], my_sample_points[:,2], my_sample_normals[:,0], my_sample_normals[:,1], my_sample_normals[:,2], length=0.002, color='b')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
set_axes_equal(ax)
# Add legend
blue_patch = plt.Line2D([0], [0], linestyle="none", c='b', marker='o')
# add legend for the normals
plt.legend([blue_patch], ['original sampled points', 'new sampled points'])
plt.title('Sampled points and normals')
plt.savefig(osp.join(data_dir, 'visualize_sampled_points_normals.png'), dpi=600)
plt.show()

# %%



