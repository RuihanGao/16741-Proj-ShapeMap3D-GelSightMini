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

import basics.sensorParams as psp
from basics.CalibData import CalibData
from utils import skewMat, fill_blank, generate_normals, processInitialFrame

# pybullet imports 
from UR5Sim import UR5Sim
import time
import math

class Sim(): 
    def __init__(self, obj_name):
        self.sim = UR5Sim(obj_name)
        self.sim.add_gui_sliders()
 
class heightMapGenerator:

    def __init__(self, data_path, ply_path, save_path, gel_map_path, calib_folder, use_pseudo_bg=False, sensor_name=""):
        """
        Initialize the heightmap generator.

        Args:
            data_path (str): path to the .mat file containing the sampled points and normals
            ply_path (str): path to the .ply file containing the dense point cloud
            save_path (str): path to the folder to save the generated heightmaps, contact masks, poses, and tactile images
            gel_map_path (str): path to the .npy file containing the gelmap
            calib_folder (str): path to the folder containing the tactile image config files
            use_pseudo_bg (bool): whether to use the pseudo background image
            sensor_name (str): name of the sensor, e.g. "_gsmini". Use this to distinguish config files (dataPack.npz, polycalib.npz) for different sensors.
        """

        # change path to cwd
        abspath = os.path.abspath(__file__)
        dname = os.path.dirname(abspath)
        os.chdir(dname)

        # meta data
        data_dict = scipy.io.loadmat(data_path)
        self.sample_points = data_dict['samplePoints'] # N+1 * 3
        self.sample_normals = data_dict['sampleNormals'] # N+1 * 3
        self.numSamples = data_dict['numSamples'][0][0] # + 1 # N maybe need to fix this
        print("Total sample poses #" + str(self.numSamples))


        # dense point cloud with vertice information
        ply_file = open(ply_path)
        lines = ply_file.readlines()
        verts_num = int(lines[3].split(' ')[-1])
        verts_lines = lines[10:10 + verts_num]
        vertices = np.array([list(map(float, l.strip().split(' '))) for l in verts_lines])
        self.all_points_hom = np.append(vertices, np.ones([len(vertices),1]),1) # homogenous coords
        print("Total .ply vertices #" + str(self.all_points_hom.shape[0]))

        if osp.exists(save_path):
            print(f"Removing existing folder {save_path}")
            os.system(f"rm -rf {save_path}")
        os.makedirs(save_path)
        self.save_folder = osp.join(save_path,'gt_height_map')
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)
        self.mask_folder = osp.join(save_path,'gt_contact_mask')
        if not os.path.exists(self.mask_folder):
            os.makedirs(self.mask_folder)
        self.tactile_folder = osp.join(save_path,'tactile_imgs')
        if not os.path.exists(self.tactile_folder):
            os.makedirs(self.tactile_folder)
        self.sim_folder = osp.join(save_path,'sim')
        if not os.path.exists(self.sim_folder):
            os.makedirs(self.sim_folder)

        self.calib_data = osp.join(calib_folder, f"polycalib{sensor_name}.npz")
        self.calib_data = CalibData(self.calib_data)
        rawData = osp.join(calib_folder, f"dataPack{sensor_name}.npz")
        data_file = np.load(rawData,allow_pickle=True)
        if use_pseudo_bg:
            self.pseudo_bg = np.load(osp.join(calib_folder, "pseudo_bg_img.npy"))
            cv2.imwrite(osp.join(save_path,'pseudo_bg.jpg'), self.pseudo_bg)
        self.bg_proc = processInitialFrame(data_file['f0'])
        # save the processed background image
        cv2.imwrite(osp.join(save_path,'bg_proc.jpg'), self.bg_proc)
        self.use_pseudo_bg = use_pseudo_bg

        ## heightmap config
        self.pose_file = open(osp.join(save_path,'pose.txt'),'w')
        self.gel_map = np.load(gel_map_path)
        self.max_g = np.max(self.gel_map) # maximum pixel value of gelmap
        self.max_z = psp.pressing_depth / 1000.0 # 1.5mm

        ## tactile image config
        bins = psp.numBins
        [xx, yy] = np.meshgrid(range(psp.w), range(psp.h))
        xf = xx.flatten()
        yf = yy.flatten()
        self.alpha = np.array([xf*xf,yf*yf,xf*yf,xf,yf,np.ones(psp.h*psp.w)]).T # [76800, 6]

        binm = bins - 1
        self.x_binr = 0.5*np.pi/binm # x [0,pi/2]
        self.y_binr = 2*np.pi/binm # y [-pi, pi]

    def points2pixels(self, pts):
        points_pix = pts * 1000.0 / psp.pixmm
        points_pix[:,1] +=  psp.h//2
        points_pix[:,0] +=  psp.w//2
        return points_pix

    # find valid points in sensor frame [-320, -240] to [320, 240]
    def findValid(self,pts):
        x_valid = (pts[:,0] * 1000.0 / psp.pixmm < psp.w//2) & (pts[:,0] * 1000.0 / psp.pixmm > -1*psp.w//2)
        y_valid = (pts[:,1] * 1000.0 / psp.pixmm < psp.h//2) & (pts[:,1] * 1000.0 / psp.pixmm > -1*psp.h//2)
        mask_valid = x_valid & y_valid
        return pts[mask_valid,:]

    # generate ground truth heightmap, contact mask, poses, and tactile images
    def generate_heightMap_and_tactileImage(self, verbose=False):

        print(f"In function generate_heightMap_and_tactileImage ....")
        # loop over all poses 
        for i in range(self.numSamples):
            if verbose: print(f"Processing pose {i} ...")

            ######## Generate ground truth heightmap, contact mask, poses from sample points
            cur_point = self.sample_points[i,:]
            cur_normal = -1*self.sample_normals[i,:]
            z_axis = np.array([0,0,1])
            if verbose: print(f"cur_point: {cur_point}, cur_normal: {cur_normal}, z_axis: {z_axis}")

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

            valid_points = self.findValid(all_points_p)
            valid_points_pix = self.points2pixels(valid_points)

            raw_map = np.zeros((psp.h,psp.w))
            raw_map[(valid_points_pix[:,1]).astype(int),(valid_points_pix[:,0]).astype(int)] = valid_points_pix[:,2]

            # avoid negative sensor readings 
            zero_gel_map =  self.gel_map - self.max_g             # find the min point
            check_positive = raw_map - zero_gel_map
            shift = np.min(check_positive) * psp.pixmm / 1000.0 # negative value, in pix
            # reset T_wp
            new_origin = np.array([0.0,0.0,shift,1.0])
            T_wp[0:3,3] = np.dot(T_wp, new_origin)[0:3]

            valid_points[:,2] -= shift # move the origin
            if verbose: 
                print(f"check range of z of valid points in mm: min {np.min(valid_points[:,2])*1000}, max {np.max(valid_points[:,2])*1000}")
            z_valid = valid_points[:,2] * 1000.0 < (psp.pressing_depth)
            valid_points = valid_points[z_valid,:]

            valid_points_pix = self.points2pixels(valid_points)
            height_map = np.zeros((psp.h,psp.w))
            height_map[(valid_points_pix[:,1]).astype(int),(valid_points_pix[:,0]).astype(int)] = -1*valid_points[:,2] + self.max_z
            height_map = fill_blank(height_map) # interpolate and fill holes 
            height_map = height_map * 1000.0 / psp.pixmm # convert to pixel
            
            heightmap_name = str(i) + '.npy'
            

            ## generate contact mask
            dome_map = -1 * self.gel_map + self.max_g
            contact_mask = height_map > dome_map
            zq = np.zeros((psp.h,psp.w))
            # use dome_map as the background for non-contact region and use height_map for contact region
            zq[contact_mask] = height_map[contact_mask]
            zq[~contact_mask] = dome_map[~contact_mask]

            # # TODO: For debugging. Force the zq to be the same as the dome_map, so no contact. Check what the sim_map could be. 
            # # Hypothesis: the sim_map should have all zeros before adding bg_proc.
            # zq = dome_map
            # contact_mask = np.zeros((psp.h,psp.w), dtype=bool)

            # contact smooth
            pressing_height_pix = psp.pressing_depth/psp.pixmm
            mask = (zq-(dome_map)) > pressing_height_pix * 0.4
            mask = mask & contact_mask

            # check the heightmap
            print("Max heightmap value (pix): " + str(np.max(-1*(height_map-self.max_z))))

            # write pose
            r = R.from_matrix(T_wp[0:3,0:3])
            qr = r.as_quat() # [x y z w] format
            t = T_wp[0:3,3]
            self.pose_file.write(heightmap_name + "," + str(qr) + "," + str(t) + "\n")

            # save GT heightmap and contact mask
            np.save(osp.join(self.save_folder,heightmap_name), zq)
            np.save(osp.join(self.mask_folder,heightmap_name), mask)

            ######## load heightmaps and generate tactile image
            print(f"Generating tactile image for pose {i} ...")
            grad_mag, grad_dir, _ = generate_normals(zq) # shape grad_mag (240, 320), grad_dir (240, 320)

            sim_img = np.zeros((psp.h,psp.w,3))
            idx_x = np.floor(grad_mag/self.x_binr).astype('int')
            idx_y = np.floor((grad_dir+np.pi)/self.y_binr).astype('int')

            params_r = self.calib_data.grad_r[idx_x,idx_y,:]
            params_r = params_r.reshape((psp.h*psp.w), params_r.shape[2])
            params_g = self.calib_data.grad_g[idx_x,idx_y,:]
            params_g = params_g.reshape((psp.h*psp.w), params_g.shape[2])
            params_b = self.calib_data.grad_b[idx_x,idx_y,:]
            params_b = params_b.reshape((psp.h*psp.w), params_b.shape[2])

            est_r = np.sum(self.alpha * params_r,axis = 1)
            est_g = np.sum(self.alpha * params_g,axis = 1)
            est_b = np.sum(self.alpha * params_b,axis = 1)
            sim_img[:,:,0] = est_r.reshape((psp.h,psp.w))
            sim_img[:,:,1] = est_g.reshape((psp.h,psp.w))
            sim_img[:,:,2] = est_b.reshape((psp.h,psp.w))
            np.save(osp.join(self.sim_folder,heightmap_name), sim_img)
            # TODO: check the sim_img for original sensor
            
            
            # write tactile image
            sim_img = sim_img + self.bg_proc
            # if self.use_pseudo_bg:
            #     sim_img = sim_img - self.pseudo_bg
            if verbose: print(f"sim_img range {np.min(sim_img):.3f} - {np.max(sim_img):.3f}")
            
            img_name = str(i) + '.jpg'
            save_path = osp.join(self.tactile_folder,img_name)
            if verbose: print(f"Saving tactile image to {save_path}")
            cv2.imwrite(save_path, sim_img)

            # if i > 3:
            #     print(f"Stopping after {i} poses for debugging ...") 
            #     break

        self.pose_file.close()

if __name__ == "__main__":
    # obj = 'power_drill'
    obj =  'mustard_bottle'
    # obj = 'ball'
    data_path = osp.join('..','..','..','data',obj,obj+'_60sampled_python.mat') # load sampled points, normals
    ply_path = osp.join('..','..','..','data',obj,obj+'.ply') # load dense point cloud
    save_path = osp.join('..','..','..','processed_data',obj)
    gel_map_path = osp.join('..', 'config', 'gelmap5.npy') # heightmap config file. Original: gelmap2.npy. Revised for GelSight mini: gelmap5.npy
    calib_folder = osp.join('..','config') # tactile img config file
    sensor_name = "_gsmini"  # "_gelsight"

    # s = Sim(obj)
    generator = heightMapGenerator(data_path, ply_path, save_path, gel_map_path, calib_folder, use_pseudo_bg=True, sensor_name=sensor_name)
    generator.generate_heightMap_and_tactileImage(verbose=True)


