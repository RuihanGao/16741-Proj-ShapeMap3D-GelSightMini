#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.
# https://github.com/facebookresearch/DeepSDF/blob/master/deep_sdf/metrics/chamfer.py

import numpy as np
from scipy.spatial import cKDTree as KDTree
import trimesh

import argparse
import logging
import json
import os
from os import path as osp
import sys 

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def compute_trimesh_chamfer(gt_points, gen_mesh, num_mesh_samples=30000):
    """
    This function computes a symmetric chamfer distance, i.e. the sum of both chamfers.
    gt_points: trimesh.points.PointCloud of just poins, sampled from the surface (see
               compute_metrics.ply for more documentation)
    gen_mesh: trimesh.base.Trimesh of output mesh from whichever autoencoding reconstruction
              method (see compute_metrics.py for more)
    """

    gen_points_sampled = trimesh.sample.sample_surface(gen_mesh, num_mesh_samples)[0]
    gt_points_np = gt_points.vertices

    # fig = plt.figure()
    # ax = plt.subplot(1, 1, 1, projection='3d')
    # # ax.set_title("Point cloud", size=10)
    # ax.scatter3D(gen_points_sampled[:, 0], gen_points_sampled[:, 1], gen_points_sampled[:, 2], c=gen_points_sampled[:, 2], cmap='viridis', s = .1)
    # ax.scatter3D(gt_points_np[:, 0], gt_points_np[:, 1], gt_points_np[:, 2], c='gray', cmap='viridis', s = .1)
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # fig.tight_layout()

    # plt.show(block=False)
    # plt.close()

    # one direction
    gen_points_kd_tree = KDTree(gen_points_sampled)
    one_distances, one_vertex_ids = gen_points_kd_tree.query(gt_points_np)
    gt_to_gen_chamfer = np.mean(np.square(one_distances))

    # other direction
    gt_points_kd_tree = KDTree(gt_points_np)
    two_distances, two_vertex_ids = gt_points_kd_tree.query(gen_points_sampled)
    gen_to_gt_chamfer = np.mean(np.square(two_distances))

    return gt_to_gen_chamfer + gen_to_gt_chamfer


def eval_object(obj):
    print("Object: " + obj)
    ground_truth_path = osp.join('..','..','..','models', obj, 'google_512k', 'nontextured.ply') # load dense point cloud
    ground_truth_points = trimesh.load(ground_truth_path)

    reconstruction_folder = osp.join('/home/suddhu/software/GPIS/results',obj, 'k=thinplate_t=incremental_g=fcrn/stl')
    files = os.listdir(reconstruction_folder)
    files = [x for x in files if any(c.isdigit() for c in x)]
    files = [os.path.splitext(x)[0] for x in files]
    files = list(map(int, files))
    files.sort()
    files = [str(x) + '.stl' for x in files]

    csv_file = osp.join('/home/suddhu/software/GPIS/results', obj, 'k=thinplate_t=incremental_g=fcrn/chamfer.csv')
    i = 1
    with open(csv_file, "w") as fname:
        fname.write("shape, chamfer_dist\n")
        for f in files:
            extension = os.path.splitext(f)[1]
            if (extension == '.stl'):
                reconstruction = trimesh.load(osp.join(reconstruction_folder, f))
                chamfer_dist = compute_trimesh_chamfer(ground_truth_points,reconstruction)
                print("chamfer #" + str(i) + " : " +  str(chamfer_dist))
                i += 1
                fname.write("{}, {}\n".format(obj, chamfer_dist))

    print("Saved as: " + csv_file)

if __name__ == "__main__":
    # change path to cwd
    abspath = osp.abspath(__file__)
    dname = osp.dirname(abspath)
    os.chdir(dname)

    objectList = ["002_master_chef_can","004_sugar_box", "005_tomato_soup_can", "010_potted_meat_can", "021_bleach_cleanser", "036_wood_block"]

    for obj in objectList: 
        eval_object(obj)
