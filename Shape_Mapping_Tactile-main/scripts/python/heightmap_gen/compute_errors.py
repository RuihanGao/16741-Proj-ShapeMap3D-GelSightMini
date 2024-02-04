#!/usr/bin/env python
# Load npy heightmaps/normals and compute the error 

import sys
import scipy.io
import numpy as np
import os
import csv
import imageio
from utils import generate_normals
import cv2
import matplotlib.pyplot as plt

def extractImages(path, files):
    imFiles=[]
    for f in files:
        extension = os.path.splitext(f)[1]
        if (extension == '.npy') or (extension == '.jpg'):
            imFiles.append(f)

    if not imFiles:
        print("Error: There are no .npy/.jpg files in %s folder"%(path))
        sys.exit(0)


    ext = os.path.splitext(imFiles[0])[1]

    # strip file names, sort numeric, and convert back to string
    imFiles = [os.path.splitext(x)[0] for x in imFiles]
    imFiles = list(map(int, imFiles))
    imFiles.sort()

    if ext == '.npy':
        images = np.zeros([480*640, len(imFiles)])
        for f in imFiles:
            currentFile= os.path.join(path,str(f) + ".npy")

            try:
                values = np.load(currentFile)
                if (values.shape[0] == 240):
                    values = cv2.resize(values, dsize=(640, 480), interpolation=cv2.INTER_CUBIC)

            except IOError:
                print("Error: can\'t find file or read data")

            else:
                images[:, f] = values.flatten('F')
    else:
        images = np.zeros([480*640, len(imFiles)*3])
        for f in imFiles:
            currentFile= os.path.join(path,str(f) + ".jpg")

            try:
                values = np.asarray(imageio.imread(currentFile))
            except IOError:
                print("Error: can\'t find file or read data")

            else:
                images[:, 3*f] = values[:, :, 0].flatten('F')
                images[:, 3*f + 1] = values[:, :, 1].flatten('F')
                images[:, 3*f + 2] = values[:, :, 1].flatten('F')
    return images

def extractNormals(path, files):
    imFiles=[]
    for f in files:
        extension = os.path.splitext(f)[1]
        if (extension == '.npy'):
            imFiles.append(f)

    if not imFiles:
        print("Error: There are no .npyfiles in %s folder"%(path))
        sys.exit(0)


    # strip file names, sort numeric, and convert back to string
    imFiles = [os.path.splitext(x)[0] for x in imFiles]
    imFiles = list(map(int, imFiles))
    imFiles.sort()

    normals = np.zeros([480*640, 3*len(imFiles)])
    for f in imFiles:
        currentFile= os.path.join(path,str(f) + ".npy")
        try:
            values = np.load(currentFile)
            if (values.shape[0] == 240):
                values = cv2.resize(values, dsize=(640, 480), interpolation=cv2.INTER_CUBIC)
        except IOError:
            print("Error: can\'t find file or read data")
        else:
            _, _, normal = generate_normals(values)
            normals[:, 3*f] = normal[:, :, 0].flatten('F')
            normals[:, 3*f + 1] = normal[:, :, 1].flatten('F')
            normals[:, 3*f + 2] = normal[:, :, 2].flatten('F')

    return normals

def readPoses(path):
    N = sum(1 for line in open(path))

    poses = np.zeros([N, 7])
    i = 0
    with open(path, 'r') as fd:
        reader = csv.reader(fd)
        for row in reader:
            quat = np.fromstring(row[1].strip("[]"), dtype=float, sep=' ')
            pos = np.fromstring(row[2].strip("[]"), dtype=float, sep=' ')
            pose = np.concatenate((pos, quat), axis=0)
            poses[i, :] = pose
            i += 1
    return poses

def compute_errors(gelsight_folder, fcrn_folder, mlp_folder, lookup_folder, contact_folder):
    heightmapPath = os.path.join(gelsight_folder, "gt_height_map")
    heightmapFiles = (os.listdir(heightmapPath))
    contactmaskPath = os.path.join(gelsight_folder, "gt_contact_mask")
    contactmaskFiles = (os.listdir(contactmaskPath))
    tactileImagePath = os.path.join(gelsight_folder, "tactile_imgs")
    tactileImageFiles = (os.listdir(tactileImagePath))

    # ground truth measurements
    gt_heightmaps = extractImages(heightmapPath, heightmapFiles)
    gt_normalmaps = extractNormals(heightmapPath, heightmapFiles)
    gt_contactmasks = extractImages(contactmaskPath, contactmaskFiles)

    # estimated contact mask
    if os.path.exists(contact_folder):
        contactMaskFiles = (os.listdir(contact_folder))
        est_contactmasks = extractImages(contact_folder, contactMaskFiles)

    ## method 1: FCRN
    if os.path.exists(fcrn_folder):
        genHeightmapPath = fcrn_folder
        genHeightmapFiles = (os.listdir(genHeightmapPath))

        fcrn_heightmaps = extractImages(genHeightmapPath, genHeightmapFiles)
        fcrn_normalmaps = extractNormals(genHeightmapPath, genHeightmapFiles)

    ## method 2: MLP     
    if os.path.exists(mlp_folder):
        genHeightmapPath = mlp_folder
        genHeightmapFiles = (os.listdir(genHeightmapPath))

        mlp_heightmaps = extractImages(genHeightmapPath, genHeightmapFiles)
        mlp_normalmaps = extractNormals(genHeightmapPath, genHeightmapFiles)

    ## method 3: lookup table 
    if os.path.exists(lookup_folder):
        genHeightmapPath = lookup_folder
        genHeightmapFiles = (os.listdir(genHeightmapPath))

        lookup_heightmaps = extractImages(genHeightmapPath, genHeightmapFiles)
        lookup_normalmaps = extractNormals(genHeightmapPath, genHeightmapFiles)
    

    # fig = plt.figure(1)
    # ax1 = fig.add_subplot(221)
    # ax1.title.set_text('GT heightmap')
    # plt.imshow(gt_heightmaps[:, 2].reshape((h, w), order='F'))

    # ax2 = fig.add_subplot(222)
    # ax2.title.set_text('FCRN heightmap')
    # plt.imshow(fcrn_heightmaps[:, 2].reshape((h, w), order='F'))

    # ax2 = fig.add_subplot(223)
    # ax2.title.set_text('MLP heightmap')
    # plt.imshow(mlp_heightmaps[:, 2].reshape((h, w), order='F'))

    # ax2 = fig.add_subplot(224)
    # ax2.title.set_text('Lookup heightmap')
    # plt.imshow(lookup_heightmaps[:, 2].reshape((h, w), order='F'))

    # plt.show()

    for im_id in range(len(heightmapFiles)):
        gt_heightmaps[:, im_id] = np.multiply(gt_heightmaps[:, im_id], gt_contactmasks[:, im_id])
        fcrn_heightmaps[:, im_id] = np.multiply(fcrn_heightmaps[:, im_id], gt_contactmasks[:, im_id])
        mlp_heightmaps[:, im_id] = np.multiply(mlp_heightmaps[:, im_id], gt_contactmasks[:, im_id])
        lookup_heightmaps[:, im_id] = np.multiply(lookup_heightmaps[:, im_id], gt_contactmasks[:, im_id])

        h = 480
        w = 640
        maxVal = np.amax(np.array([gt_heightmaps[:, im_id].max(), fcrn_heightmaps[:, im_id].max(), mlp_heightmaps[:, im_id].max(), lookup_heightmaps[:, im_id].max()]))
        minVal = np.amin(np.array([gt_heightmaps[:, im_id].min(), fcrn_heightmaps[:, im_id].min(), mlp_heightmaps[:, im_id].min(), lookup_heightmaps[:, im_id].min()]))

        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8.5, 5))

        sAxes = axes.flatten()

        sAxes[0].title.set_text('GT heightmap')
        sAxes[0].set_axis_off()
        gt_img = gt_heightmaps[:, im_id].reshape((h, w), order='F')
        gt_img = np.ma.array(gt_img, mask=(gt_img == 0))
        sAxes[0].imshow(gt_img, cmap='viridis', vmin=minVal, vmax=maxVal)
        
        
        sAxes[1].title.set_text('FCRN heightmap')
        sAxes[1].set_axis_off()
        fcrn_img = fcrn_heightmaps[:, im_id].reshape((h, w), order='F')
        fcrn_img = np.ma.array(fcrn_img, mask=(fcrn_img == 0))
        sAxes[1].imshow(fcrn_img, cmap='viridis', vmin=minVal, vmax=maxVal)
        
        
        sAxes[2].title.set_text('MLP heightmap')
        sAxes[2].set_axis_off()
        mlp_img = mlp_heightmaps[:, im_id].reshape((h, w), order='F')
        mlp_img = np.ma.array(mlp_img, mask=(mlp_img == 0))
        sAxes[2].imshow(mlp_img, cmap='viridis', vmin=minVal, vmax=maxVal)

        sAxes[3].title.set_text('Lookup heightmap')
        sAxes[3].set_axis_off()
        lookup_img = lookup_heightmaps[:, im_id].reshape((h, w), order='F')
        lookup_img = np.ma.array(lookup_img, mask=(lookup_img == 0))
        im = sAxes[3].imshow(lookup_img, cmap='viridis', vmin=minVal, vmax=maxVal)

        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.95)
        cbar.set_ticks(np.arange(minVal, maxVal + 1, (minVal + maxVal)/2.0))
        cbar.set_ticklabels(['low', 'medium', 'high'])

        plt.show()

    # ax3 = fig.add_subplot(325)
    # ax3.title.set_text('Error map')
    # plt.imshow(np.abs(error))

    # ax = fig.add_subplot(1, 2, 2, projection='3d')
    # ax.title.set_text(message)
    # [xq, yq] = np.meshgrid(range(0, psp.w, 1), range(0, psp.h, 1))
    # # plt.gca().set_zlim(bottom=0.2)
    # ax.scatter(xq[::5,::5], yq[::5,::5], flip_height_map[::5,::5]  , marker='.', color='r', label='Estimated height map') # flip and convert to mm
    # ax.scatter(xq[::5,::5], yq[::5,::5], flip_gt[::5,::5] , marker='.', color='b', label='Ground truth height map') # flip and convert to mm
    # ax.set_xlabel('X Label')
    # ax.set_ylabel('Y Label')
    # ax.set_zlabel('Z Label')
    # ax.legend()
    # plt.show()

    # compute error
    error = (gt_heightmaps - fcrn_heightmaps)
    mean_error = np.mean(np.abs(error))
    # all_err += mean_error
    # mean_error_contact = np.mean(np.abs(error[gt_mask]))
    # all_contact_err += mean_error_contact

    return

if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(0)

    # change path to cwd
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
    obj = str(sys.argv[1])
    touchFile = str(sys.argv[2])
    print(obj)

    gelsight_folder = os.path.join('..','..','..','gelsight_data', os.path.splitext(touchFile)[0], obj)
    fcrn_folder = os.path.join('..','..','..','generated_data', os.path.splitext(touchFile)[0], 'fcrn', obj)
    mlp_folder = os.path.join('..','..','..','generated_data', os.path.splitext(touchFile)[0], 'mlp', obj)
    lookup_folder = os.path.join('..','..','..','generated_data', os.path.splitext(touchFile)[0], 'lookup', obj)
    contact_folder = os.path.join('..','..','..','generated_data', os.path.splitext(touchFile)[0], 'contact_masks', obj)

    compute_errors(gelsight_folder, fcrn_folder, mlp_folder, lookup_folder, contact_folder)
