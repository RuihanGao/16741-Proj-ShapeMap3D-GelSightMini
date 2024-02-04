import matplotlib.pyplot as plt
import numpy as np
import os
from os import path as osp
import cv2
import scipy
from scipy import io
from scipy.ndimage import correlate
import scipy.ndimage as ndimage
from scipy import interpolate
from scipy.spatial.transform import Rotation as R

import sys
sys.path.append("..")
import basics.sensorParams as psp

def skewMat(v):
    mat = np.zeros((3,3))
    mat[0,1] = -1*v[2]
    mat[0,2] = v[1]

    mat[1,0] = v[2]
    mat[1,2] = -1*v[0]

    mat[2,0] = -1*v[1]
    mat[2,1] = v[0]

    return mat

def fill_blank(img):
    # here we assume there are some zero value holes in the image,
    # and we hope to fill these holes with interpolation
    x = np.arange(0, img.shape[1])
    y = np.arange(0, img.shape[0])
    #mask invalid values
    array = np.ma.masked_where(img == 0, img)
    xx, yy = np.meshgrid(x, y)
    #get only the valid values
    x1 = xx[~array.mask]
    y1 = yy[~array.mask]
    newarr = img[~array.mask]

    GD1 = interpolate.griddata((x1, y1), newarr.ravel(),
                              (xx, yy),
                                 method='cubic', fill_value = 0) # cubic # nearest # linear
    return GD1

def fill_blank_spline(img):
    # here we assume there are some zero value holes in the image,
    # and we hope to fill these holes with interpolation
    x = np.arange(0, img.shape[1])
    y = np.arange(0, img.shape[0])
    #mask invalid values
    array = np.ma.masked_where(img == 0, img)
    xx, yy = np.meshgrid(x, y)
    #get only the valid values
    x1 = xx[~array.mask]
    y1 = yy[~array.mask]
    newarr = img[~array.mask]

    tck = interpolate.bisplrep(x1, y1, newarr, s=0)
    znew = interpolate.bisplev(xx, yy, tck)
    plt.imshow(znew)
    plt.show()
    return znew

# obj = 'bunny'
obj = 'mustard_bottle'
# obj = 'power_drill'
data_path = osp.join('..', '..','..','..','data',obj,obj+'_50sampled.mat')
data_dict = scipy.io.loadmat(data_path)
print(data_dict['vertices'].shape)
print(data_dict['samplePoints'].shape)
print(data_dict['sampleNormals'].shape)
print(data_dict['numSamples'])
print(data_dict['faces'].shape)
print(data_dict['normals'].shape)
# pp = input("pause")

all_points = data_dict['vertices'] # 34817 * 3
sample_points = data_dict['samplePoints'] # 51 * 3
sameple_normals = data_dict['sampleNormals'] # 51 * 3
numSamples = data_dict['numSamples'][0][0] # 50
# for i in range(numSamples+1):
#     print(sample_points[i,:])
print("range for x")
print(np.max(all_points[:,0]))
print(np.min(all_points[:,0]))

print("range for y")
print(np.max(all_points[:,1]))
print(np.min(all_points[:,1]))

print("range for z")
print(np.max(all_points[:,2]))
print(np.min(all_points[:,2]))

all_points_hom = np.append(all_points, np.ones([len(all_points),1]),1)
print(all_points_hom.shape)

# # read in ply
ply_path = osp.join('..', '..','..','..','processed_data',obj,obj+'.ply')
f = open(ply_path)
lines = f.readlines()
verts_num = int(lines[3].split(' ')[-1])
# self.faces_num = int(lines[7].split(' ')[-1])
verts_lines = lines[10:10 + verts_num]
# faces_lines = lines[10 + self.verts_num:]
vertices = np.array([list(map(float, l.strip().split(' '))) for l in verts_lines])
# self.faces = np.array([list(map(int, l.strip().split(' '))) for l in faces_lines])[:,1:]
print("the shape for vertices is " + str(vertices.shape))

pp = input("pause")

all_points_hom = np.append(vertices, np.ones([len(vertices),1]),1)
print(all_points_hom.shape)
#
# # test with few
# numSamples = 3

## save genertated height map
SAVE = False

if SAVE:
    save_path = osp.join('..','..','..','processed_data',obj)
    save_folder = osp.join(save_path,'gt_height_map')
    pose_file = open(osp.join(save_path,'pose.txt'),'w')

# test certain samples
# sample_idx = [8,15,16,17,19,22,27,40,46,48,50]
sample_idx = [8,15]

for i in sample_idx:
# for i in range(numSamples+1):
    cur_point = sample_points[i,:]
    cur_normal = -1*sameple_normals[i,:]
    z_axis = np.array([0,0,1])
    # nx = cur_normal[0]
    # ny = cur_normal[1]
    # nz = cur_normal[2]

    T_wp = np.zeros((4,4)) # transform from point coord to world coord
    T_wp[3,3] = 1
    T_wp[0:3,3] = cur_point # t

    if (z_axis==cur_normal).all():
        T_wp[0:3,0:3] = np.indentity(3)
    else:
        v = np.cross(z_axis,cur_normal)
        s = np.linalg.norm(v)
        c = np.dot(z_axis,cur_normal)
        Rot = np.identity(3) + skewMat(v) + np.linalg.matrix_power(skewMat(v),2) * (1-c)/(s**2)
        T_wp[0:3,0:3] = Rot
    #
    # T_wp = np.zeros((4,4)) # transform from point coord to world coord
    # T_wp[3,3] = 1
    # T_wp[0:3,3] = cur_point # t
    # T_wp[0:3,2] = cur_normal # R[2]
    #
    # if nx != 0:
    #     R21 = np.random.random_sample()
    #     R31 = np.random.random_sample()
    #     R11 = -1*(R21*cur_normal[1] + R31*cur_normal[2])/nx
    # elif ny != 0:
    #     R11 = np.random.random_sample()
    #     R31 = np.random.random_sample()
    #     R11 = -1*(R11*cur_normal[0] + R31*cur_normal[2])/ny
    # else:
    #     R11 = np.random.random_sample()
    #     R21 = np.random.random_sample()
    #     R31 = -1*(R11*cur_normal[0] + R21*cur_normal[1])/nz
    #
    # R1 = np.array([R11, R21, R31])
    # norm_R1 = np.linalg.norm(R1)
    # R1 = R1/norm_R1
    # T_wp[0:3,0] = R1 # R[0]
    # R2 = np.cross(R1, cur_normal)
    # T_wp[0:3,1] = R2 # R[1]

    # ## test R
    # R = T_wp[0:3,0:3]
    # print(np.dot(R.T,R))
    T_pw = np.linalg.inv(T_wp) # transform from world coord to point coord
    all_points_p = np.dot(T_pw, all_points_hom.T).T[:,0:3] # origin is the center of image plane, N * 3
    # print(all_points_p.shape)
    # test_point = np.array([cur_point[0], cur_point[1], cur_point[2], 1])
    # print(np.dot(T_pw,test_point))

    print(np.min(all_points_p[:,2]), np.max(all_points_p[:,2]))
    # print(np.min(all_points_p[:,2]))
    z_valid = (all_points_p[:,2] * 1000.0 < psp.pressing_depth) & (all_points_p[:,2] * 1000.0 > -1*psp.pressing_depth)
    x_valid = (all_points_p[:,0] * 1000.0 / psp.pixmm < psp.w//2) & (all_points_p[:,0] * 1000.0 / psp.pixmm > -1*psp.w//2)
    y_valid = (all_points_p[:,1] * 1000.0 / psp.pixmm < psp.h//2) & (all_points_p[:,1] * 1000.0 / psp.pixmm > -1*psp.h//2)
    mask_valid = z_valid & x_valid & y_valid
    valid_points = all_points_p[mask_valid,:]
    print(valid_points.shape)

    ###### visualization ######
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(valid_points[:,0], valid_points[:,1], -1*valid_points[:,2]+np.max(valid_points[:,2]), marker='o')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()
    print("before interpolate, min, max")
    print(np.min(-1*valid_points[:,2]+np.max(valid_points[:,2])))
    print(np.max(-1*valid_points[:,2]+np.max(valid_points[:,2])))
    ###### visualization end ######

    valid_points_pix = valid_points * 1000.0 / psp.pixmm
    # print(np.min(valid_points_pix[:,1]))
    # print(np.max(valid_points_pix[:,1]))
    valid_points_pix[:,1] += 240
    valid_points_pix[:,0] += 320
    height_map = np.zeros((psp.h,psp.w))
    min_z = np.min(valid_points[:,2])
    max_z = np.max(valid_points[:,2])
    height_map[(valid_points_pix[:,1]).astype(int),(valid_points_pix[:,0]).astype(int)] = -1*valid_points[:,2] + max_z
    # # unique the map, slow!
    # # mask_surf = np.zeros((valid_points_pix.shape[0]))
    # for k in range(valid_points_pix.shape[0]):
    #     val = height_map[(valid_points_pix[k,1]).astype(int),(valid_points_pix[k,0]).astype(int)]
    #     # if val != 0:
    #     #     print("here")
    #     height_map[(valid_points_pix[k,1]).astype(int),(valid_points_pix[k,0]).astype(int)] = max(-1*valid_points[k,2] + max_z, val)
    #     # else:
    #     #     height_map[(valid_points_pix[k,1]).astype(int),(valid_points_pix[k,0]).astype(int)] = -1*valid_points[k,2] + max_z
    #
    # plt.imshow(height_map)
    # plt.show()
    height_map = fill_blank(height_map)
    print("after interpolate, min, max")
    print(np.min(height_map))
    print(np.max(height_map))
    plt.imshow(height_map)
    plt.show()

    height_map = height_map * 1000.0 / psp.pixmm
    print("after rescale, min, max")
    print(np.min(height_map))
    print(np.max(height_map))
    # height_map = fill_blank_spline(height_map)
    # tck = interpolate.bisplrep(valid_points_pix[:,0].astype(int), valid_points_pix[:,1].astype(int), -1*valid_points_pix[:,2] + max_z)
    # x = np.arange(0, height_map.shape[1])
    # y = np.arange(0, height_map.shape[0])
    # xx, yy = np.meshgrid(x, y)
    x = np.arange(psp.w)
    y = np.arange(psp.h)
    xx, yy = np.meshgrid(x, y)
    # znew = interpolate.bisplev(y, x, tck)
    # plt.imshow(znew)
    # plt.show()
    # plt.imshow(height_map)
    # plt.show()

    # test after interpolation
    # print(x.shape)
    # print(y.shape)
    # print(znew.shape)
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.scatter(yy, xx, height_map, marker='o')
    # ax.set_xlabel('X Label')
    # ax.set_ylabel('Y Label')
    # ax.set_zlabel('Z Label')
    # plt.show()

    if SAVE:
        # write files!
        r = R.from_matrix(T_wp[0:3,0:3])
        qr = r.as_quat() # (x, y, z, w) 
        t = T_wp[0:3,3]
        img_name = str(i)+'.npy'
        pose_file.write(img_name + "," + str(qr) + "," + str(t) + "\n")
        np.save(osp.join(save_folder,img_name), height_map)

if SAVE:
    pose_file.close()
