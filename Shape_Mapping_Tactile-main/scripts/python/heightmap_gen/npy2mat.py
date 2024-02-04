#!/usr/bin/env python
# https://github.com/ruitome/npy_to_matlab/blob/master/npy_to_matlab.py

import sys
import scipy.io
import numpy as np
import os
import csv
import imageio
from utils import generate_normals
import cv2
import matplotlib.pyplot as plt

def extractDepthMap(files):
    try:
        pc = np.load(files)
    except IOError:
        print("Error: can\'t find file or read data")
    return pc

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
                    # fig = plt.figure(1)
                    # ax1 = fig.add_subplot(131)
                    # ax1.title.set_text('low')
                    # plt.imshow(values)

                    values = cv2.resize(values, dsize=(640, 480), interpolation=cv2.INTER_CUBIC)

                    # ax2 = fig.add_subplot(132)
                    # ax2.title.set_text('high')
                    # plt.imshow(values)
                    # ax3 = fig.add_subplot(132)
                    # ax3.title.set_text('gt')
                    # plt.imshow(values)
                    # plt.show()


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
                images[:, 3*f + 2] = values[:, :, 2].flatten('F')
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

def npy_to_matlab(gelsight_folder, fcrn_folder, mlp_folder, lookup_folder, contact_folder):
    heightmapPath = os.path.join(gelsight_folder, "gt_height_map")
    heightmapFiles = (os.listdir(heightmapPath))
    contactmaskPath = os.path.join(gelsight_folder, "gt_contact_mask")
    contactmaskFiles = (os.listdir(contactmaskPath))
    tactileImagePath = os.path.join(gelsight_folder, "tactile_imgs")
    tactileImageFiles = (os.listdir(tactileImagePath))
    posesPath = os.path.join(gelsight_folder, "pose.txt")

    depthPath = os.path.join(gelsight_folder, "depthCam.npy")

    matStructure = {}

    matStructure['gt_heightmaps'] = extractImages(heightmapPath, heightmapFiles)
    matStructure['gt_normalmaps'] = extractNormals(heightmapPath, heightmapFiles)

    matStructure['gt_contactmasks'] = extractImages(contactmaskPath, contactmaskFiles)
    matStructure['tactileimages'] = extractImages(tactileImagePath, tactileImageFiles)

    matStructure['poses'] = readPoses(posesPath)
    matStructure['depth_map'] = extractDepthMap(depthPath)

    if os.path.exists(contact_folder):
        contactMaskFiles = (os.listdir(contact_folder))
        matStructure['est_contactmasks'] = extractImages(contact_folder, contactMaskFiles)

    filename = "gelsight_data" + '.mat'
    savePath = os.path.join(gelsight_folder, filename)
    if matStructure:
        scipy.io.savemat(savePath, matStructure)

    print(savePath)

    if os.path.exists(fcrn_folder):
        matStructure = {}
        genHeightmapPath = fcrn_folder
        genHeightmapFiles = (os.listdir(genHeightmapPath))

        matStructure['fcrn_heightmaps'] = extractImages(genHeightmapPath, genHeightmapFiles)
        matStructure['fcrn_normalmaps'] = extractNormals(genHeightmapPath, genHeightmapFiles)
        
        filename = "fcrn_data" + '.mat'
        savePath = os.path.join(fcrn_folder, filename)
        if matStructure:
            scipy.io.savemat(savePath, matStructure)
        print(savePath)

    if os.path.exists(mlp_folder):
        matStructure = {}
        genHeightmapPath = mlp_folder
        genHeightmapFiles = (os.listdir(genHeightmapPath))

        matStructure['mlp_heightmaps'] = extractImages(genHeightmapPath, genHeightmapFiles)
        matStructure['mlp_normalmaps'] = extractNormals(genHeightmapPath, genHeightmapFiles)

        filename = "mlp_data" + '.mat'
        savePath = os.path.join(mlp_folder, filename)
        if matStructure:
            scipy.io.savemat(savePath, matStructure)
        print(savePath)

    if os.path.exists(lookup_folder):
        matStructure = {}
        genHeightmapPath = lookup_folder
        genHeightmapFiles = (os.listdir(genHeightmapPath))

        matStructure['lookup_heightmaps'] = extractImages(genHeightmapPath, genHeightmapFiles)
        matStructure['lookup_normalmaps'] = extractNormals(genHeightmapPath, genHeightmapFiles)

        filename = "lookup_data" + '.mat'
        savePath = os.path.join(lookup_folder, filename)
        if matStructure:
            scipy.io.savemat(savePath, matStructure)
        print(savePath)

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

    npy_to_matlab(gelsight_folder, fcrn_folder, mlp_folder, lookup_folder, contact_folder)
