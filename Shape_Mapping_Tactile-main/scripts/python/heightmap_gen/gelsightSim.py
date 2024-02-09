# input the set of poses and object file, and generate the ground truth heightmaps, tactile images and contact masks
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

from npy2mat import *
import basics.sensorParams as psp
from basics.CalibData import CalibData
from utils import skewMat, fill_blank, generate_normals, processInitialFrame

class gelsightSim:

    def __init__(self, data_path, ply_path):
        # meta data
        data_dict = scipy.io.loadmat(data_path)
        self.sample_points = data_dict['samplePoints'] # N+1 * 3
        self.sample_normals = data_dict['sampleNormals'] # N+1 * 3
        self.numSamples = data_dict['numSamples'][0][0] # N maybe need to fix this
        print("Total sample poses #" + str(self.numSamples))


        # dense point cloud with vertice information
        ply_file = open(ply_path)
        lines = ply_file.readlines()
        verts_num = int(lines[3].split(' ')[-1])
        verts_lines = lines[10:10 + verts_num]
        vertices = np.array([list(map(float, l.strip().split(' '))) for l in verts_lines])
        vertices = vertices[:, 0:3]
        self.all_points_hom = np.append(vertices, np.ones([len(vertices),1]),1) # homogenous coords
        print("Total .ply vertices #" + str(self.all_points_hom.shape[0]))

    def local_points2pixels(self, points):
        points_pix = points * 1000.0 / psp.pixmm
        points_pix[:,1] +=  psp.h//2
        points_pix[:,0] +=  psp.w//2
        return points_pix

    # generate ground truth heightmap, contact mask, poses, and tactile images
    def generate_heightMap_and_tactileImage(self, save_path):

        calib_folder = osp.join('..','..','..','calib') # tactile img config file
        gel_map_path = osp.join(calib_folder, 'gelmap2.npy') # heightmap config file
        bg = np.load(osp.join(calib_folder, 'real_bg.npy'))

        save_folder = osp.join(save_path,'gt_height_map')
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        mask_folder = osp.join(save_path,'gt_contact_mask')
        if not os.path.exists(mask_folder):
            os.makedirs(mask_folder)
        tactile_folder = osp.join(save_path,'tactile_imgs')
        if not os.path.exists(tactile_folder):
            os.makedirs(tactile_folder)

        ## heightmap config
        pose_file = open(osp.join(save_path,'pose.txt'),'w')
        gel_map = np.load(gel_map_path)
        gel_map = cv2.GaussianBlur(gel_map.astype(np.float32),(5,5),0)
        max_g = np.max(gel_map) # maximum pixel value of gelmap
        max_z = psp.pressing_depth / 1000.0 # 1.5mm

        ## tactile image config
        bins = psp.numBins
        [xx, yy] = np.meshgrid(range(psp.w), range(psp.h))
        xf = xx.flatten()
        yf = yy.flatten()
        A = np.array([xf*xf,yf*yf,xf*yf,xf,yf,np.ones(psp.h*psp.w)]).T

        binm = bins - 1
        x_binr = 0.5*np.pi/binm # x [0,pi/2]
        y_binr = 2*np.pi/binm # y [-pi, pi]

        calib_data = osp.join(calib_folder, "polycalib.npz")
        self.calib_data = CalibData(calib_data)
        rawData = osp.join(calib_folder, "dataPack.npz")
        data_file = np.load(rawData,allow_pickle=True)
        self.bg_proc = processInitialFrame(data_file['f0'])

        # loop over all poses
        for i in range(self.numSamples):

            ######## Generate ground truth heightmap, contact mask, poses from sample points
            cur_point = self.sample_points[i,:]
            cur_normal = -1*self.sample_normals[i,:]
            z_axis = np.array([0,0,1])

            T_wp = np.zeros((4,4)) # transform from point coord to world coord
            T_wp[3,3] = 1
            T_wp[0:3,3] = cur_point # t

            if (z_axis==cur_normal).all():
                T_wp[0:3,0:3] = np.indentity(3)
            else:
                # generate rotation for sensor
                v = np.cross(z_axis,cur_normal)
                s = np.linalg.norm(v)
                c = np.dot(z_axis,cur_normal)
                Rot = np.identity(3) + skewMat(v) + np.linalg.matrix_power(skewMat(v),2) * (1-c)/(s**2) # rodrigues
                T_wp[0:3,0:3] = Rot

            T_pw = np.linalg.inv(T_wp) # transform from world coord to point coord
            all_points_p = np.dot(T_pw, self.all_points_hom.T).T[:,0:3] # origin is the center of image plane, N * 3

            # z_valid = (all_points_p[:,2] * 1000.0 < psp.pressing_depth)

            # find valid points in sensor frame [-320, -240] to [320, 240]
            x_valid = (all_points_p[:,0] * 1000.0 / psp.pixmm < psp.w//2) & (all_points_p[:,0] * 1000.0 / psp.pixmm > -1*psp.w//2)
            y_valid = (all_points_p[:,1] * 1000.0 / psp.pixmm < psp.h//2) & (all_points_p[:,1] * 1000.0 / psp.pixmm > -1*psp.h//2)
            mask_valid = x_valid & y_valid
            valid_points = all_points_p[mask_valid,:]

            valid_points_pix = self.local_points2pixels(valid_points)
            raw_map = np.zeros((psp.h,psp.w))
            raw_map[(valid_points_pix[:,1]).astype(int),(valid_points_pix[:,0]).astype(int)] = valid_points_pix[:,2]

            # avoid negative sensor readings
            zero_gel_map =  gel_map - max_g             # find the min point
            check_positive = raw_map - zero_gel_map
            shift = np.min(check_positive) * psp.pixmm / 1000.0 # negative value, in pix
            # reset T_wp
            new_origin = np.array([0.0,0.0,shift,1.0])
            T_wp[0:3,3] = np.dot(T_wp, new_origin)[0:3]

            valid_points[:,2] -= shift # move the origin
            z_valid = valid_points[:,2] * 1000.0 < (psp.pressing_depth)
            valid_points = valid_points[z_valid,:]

            valid_points_pix = self.local_points2pixels(valid_points)
            height_map = np.zeros((psp.h,psp.w))
            height_map[(valid_points_pix[:,1]).astype(int),(valid_points_pix[:,0]).astype(int)] = -1*valid_points[:,2] + max_z
            # ## debug ##
            # if (i > 106):
            #     print("min z value " + str(np.min(valid_points[:,2])))
            #     print("max z value " + str(np.max(valid_points[:,2])))
            #     print("min height " + str(np.min(height_map)))
            #     print("max height " + str(np.max(height_map)))
            #     filled_height_map = fill_blank(height_map)
            #     print("min height " + str(np.min(filled_height_map)))
            #     print("max height " + str(np.max(filled_height_map)))
            #     plt.imshow(height_map)
            #     plt.show()
            #     plt.imshow(filled_height_map)
            #     plt.show()
            #     p = input("pause")
            try:
                height_map = fill_blank(height_map) # interpolate and fill holes
            except:
                print("heightmap filling failed!\n")
            height_map = height_map * 1000.0 / psp.pixmm # convert to pixel

            ## generate contact mask
            dome_map = -1 * gel_map + max_g
            contact_mask = height_map > dome_map
            zq = np.zeros((psp.h,psp.w))
            zq[contact_mask] = height_map[contact_mask]
            zq[~contact_mask] = dome_map[~contact_mask]
            # contact smooth
            pressing_height_pix = psp.pressing_depth/psp.pixmm
            mask = (zq-(dome_map)) > pressing_height_pix * 0.4
            mask = mask & contact_mask

            zq_back = zq.copy()
            kernel_size = [51,31,21,11,5]
            for k in range(len(kernel_size)):
                zq = cv2.GaussianBlur(zq.astype(np.float32),(kernel_size[k],kernel_size[k]),0)
                zq[mask] = zq_back[mask]
            zq = cv2.GaussianBlur(zq.astype(np.float32),(5,5),0)
            # plt.imshow(zq)
            # plt.show()
            # p = input("pause")

            # check the heightmap
            # print("Max heightmap value (pix): " + str(np.max(-1*(height_map-max_z))))

            # write pose
            heightmap_name = str(i) + '.npy'
            r = R.from_matrix(T_wp[0:3,0:3])
            qr = r.as_quat() # [x y z w] format
            t = T_wp[0:3,3]
            pose_file.write(heightmap_name + "," + str(qr) + "," + str(t) + "\n")

            # save GT heightmap and contact mask
            np.save(osp.join(save_folder,heightmap_name), zq)
            np.save(osp.join(mask_folder,heightmap_name), mask)

            ######## load heightmaps and generate tactile image
            grad_mag, grad_dir, _ = generate_normals(zq)

            sim_img_r = np.zeros((psp.h,psp.w,3))
            idx_x = np.floor(grad_mag/x_binr).astype('int')
            idx_y = np.floor((grad_dir+np.pi)/y_binr).astype('int')

            params_r = self.calib_data.grad_r[idx_x,idx_y,:]
            params_r = params_r.reshape((psp.h*psp.w), params_r.shape[2])
            params_g = self.calib_data.grad_g[idx_x,idx_y,:]
            params_g = params_g.reshape((psp.h*psp.w), params_g.shape[2])
            params_b = self.calib_data.grad_b[idx_x,idx_y,:]
            params_b = params_b.reshape((psp.h*psp.w), params_b.shape[2])

            est_r = np.sum(A * params_r,axis = 1)
            est_g = np.sum(A * params_g,axis = 1)
            est_b = np.sum(A * params_b,axis = 1)
            sim_img_r[:,:,0] = est_r.reshape((psp.h,psp.w))
            sim_img_r[:,:,1] = est_g.reshape((psp.h,psp.w))
            sim_img_r[:,:,2] = est_b.reshape((psp.h,psp.w))

            # write tactile image
            sim_img = sim_img_r + bg
            img_name = str(i) + '.jpg'
            cv2.imwrite(osp.join(tactile_folder,img_name), sim_img)

        pose_file.close()

if __name__ == "__main__":
    if len(sys.argv) == 3:
        obj = str(sys.argv[1]) # 'mustard_bottle', 'power_drill'
        touchFile = str(sys.argv[2]) # 'mustard_bottle', 'power_drill'
    else:
        obj = "021_bleach_cleanser"
        touchFile = "textured_50sampled.mat"

    # change path to cwd
    abspath = osp.abspath(__file__)
    dname = osp.dirname(abspath)
    os.chdir(dname)

    touchFile = "textured_120sampled.mat"
    objects = sorted(os.listdir(osp.join('..','..','..','models')))
    print(len(objects))
    for obj in objects:
        # print(obj)
        if obj == ".DS_Store":
            continue
        if obj == "dimensions.txt":
            continue
        if obj == "misc":
            continue
        if obj == "README.md":
            continue

        print(obj)
        data_path = osp.join('..','..','..','models', obj, touchFile) # load sampled points, normals
        ply_path = osp.join('..','..','..','models', obj, 'google_512k', 'nontextured.ply') # load dense point cloud
        save_path = osp.join('..','..','..','gelsight_data', os.path.splitext(touchFile)[0], obj)

        generator = gelsightSim(data_path, ply_path)
        generator.generate_heightMap_and_tactileImage(save_path)
