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

# test a simple case
# cur_point = np.array([2,3,4])
# cur_normal = np.array([0,1,0])
# test_point = cur_point + cur_normal
#
# nx = cur_normal[0]
# ny = cur_normal[1]
# nz = cur_normal[2]
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
#     R21 = -1*(R11*cur_normal[0] + R31*cur_normal[2])/ny
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
#
# # ## test R
# print(T_wp[0:3,0:3])
# # R = T_wp[0:3,0:3]
# # print(np.dot(R.T,R))
# test_point_hom = np.array([test_point[0],test_point[1],test_point[2],1])
# T_pw = np.linalg.inv(T_wp) # transform from world coord to point coord
# test_point_new = np.dot(T_pw, test_point_hom.T).T[0:3] # origin is the center of image plane, N * 3
# print(test_point_new)


def skewMat(v):
    mat = np.zeros((3,3))
    mat[0,1] = -1*v[2]
    mat[0,2] = v[1]

    mat[1,0] = v[2]
    mat[1,2] = -1*v[0]

    mat[2,0] = -1*v[1]
    mat[2,1] = v[0]

    return mat

### new method
# a = np.array([0,0,1]) # old z-axis
# b = np.array([0,1,0]) # new z-axis
# v = np.cross(a,b)
# print(v)
# s = np.linalg.norm(v)
# c = np.dot(a,b)
# print(s)
# print(c)
# R = np.identity(3) + skewMat(v) + np.linalg.matrix_power(skewMat(v),2) * (1-c)/(s**2)
# print(R)


cur_point = np.array([2,3,4])
z_axis = np.array([0,0,1])
cur_normal = np.array([1,1,0])
test_point = cur_point + cur_normal

T_wp = np.zeros((4,4)) # transform from point coord to world coord
T_wp[3,3] = 1
T_wp[0:3,3] = cur_point # t

if (z_axis==cur_normal).all():
    T_wp[0:3,0:3] = np.indentity(3)
else:
    v = np.cross(z_axis,cur_normal)
    s = np.linalg.norm(v)
    c = np.dot(z_axis,cur_normal)
    R = np.identity(3) + skewMat(v) + np.linalg.matrix_power(skewMat(v),2) * (1-c)/(s**2)
    T_wp[0:3,0:3] = R

# ## test R
print(T_wp[0:3,0:3])
# R = T_wp[0:3,0:3]
# print(np.dot(R.T,R))
test_point_hom = np.array([test_point[0],test_point[1],test_point[2],1])
T_pw = np.linalg.inv(T_wp) # transform from world coord to point coord
test_point_new = np.dot(T_pw, test_point_hom.T).T[0:3] # origin is the center of image plane, N * 3
print(test_point_new)
