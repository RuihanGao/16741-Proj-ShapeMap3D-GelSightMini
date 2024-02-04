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
import pdb
import json 

def extractDepthMap(folder):

    files = os.listdir(folder)
    try:
        pc = np.load(os.path.join(folder, files[0]))
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

    if ext == '.npy':
        images = np.zeros([480*640, len(imFiles)])
        idx = 0
        for f in imFiles:
            currentFile= os.path.join(path,str(f))

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
                images[:, idx] = values.flatten('F')
                idx += 1
    else:
        images = np.zeros([480*640, len(imFiles)*3])
        idx = 0
        for f in imFiles:
            currentFile= os.path.join(path,str(f))

            try:
                values = np.asarray(imageio.imread(currentFile))
            except IOError:
                print("Error: can\'t find file or read data")
            else:
                images[:, 3*idx] = values[:, :, 0].flatten('F')
                images[:, 3*idx + 1] = values[:, :, 1].flatten('F')
                images[:, 3*idx + 2] = values[:, :, 2].flatten('F')
                idx += 1
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


    normals = np.zeros([480*640, 3*len(imFiles)])
    idx = 0
    for f in imFiles:
        currentFile= os.path.join(path,str(f))
        try:
            values = np.load(currentFile)
            if (values.shape[0] == 240):
                values = cv2.resize(values, dsize=(640, 480), interpolation=cv2.INTER_CUBIC)
        except IOError:
            print("Error: can\'t find file or read data")
        else:
            _, _, normal = generate_normals(values)
            normals[:, 3*idx] = normal[:, :, 0].flatten('F')
            normals[:, 3*idx + 1] = normal[:, :, 1].flatten('F')
            normals[:, 3*idx + 2] = normal[:, :, 2].flatten('F')
            idx += 1

    return normals

def readPoses(path):
    N = sum(1 for line in open(path)) - 1

    poses = np.zeros([N, 7])
    i = 0
    with open(path, 'r') as fd:
        reader = csv.reader(fd)
        next(reader) # skip first line 
        
        for row in reader:
            row = np.array(row, dtype=float)
            quat = row[4:]
            pos = row[1:4]
            pose = np.concatenate((pos, quat), axis=0)
            poses[i, :] = pose
            i += 1
    return poses

def readTransforms(path):
    with open(path, "r") as read_file:
        data = json.load(read_file)
    
    gripper2gelsight = data['gripper2gelsight']
    world2azure = data['world2azure']
    world2object = data['world2object']
    return gripper2gelsight, world2azure, world2object

def npy_to_matlab(base_folder, gelsight_folder, fcrn_folder, mlp_folder, lookup_folder, contact_folder):
    tactileImageFiles = (os.listdir(gelsight_folder))
    posesPath = os.path.join(base_folder, "robot.csv")
    tfPath = os.path.join(base_folder, "tf.json")

    depthPath = os.path.join(base_folder, "pc")

    matStructure = {}
    matStructure['tactileimages'] = extractImages(gelsight_folder, tactileImageFiles)
    matStructure['poses'] = readPoses(posesPath)
    matStructure['gripper2gelsight'], matStructure['world2azure'], matStructure['world2object'] = readTransforms(tfPath)
    
    matStructure['depth_map'] = extractDepthMap(depthPath)

    if os.path.exists(contact_folder):
        print('Saving est. contactmasks')
        contactMaskFiles = (os.listdir(contact_folder))
        contactMaskFiles = [x for x in contactMaskFiles if any(c.isdigit() for c in x)]

        # strip file names, sort numeric, and convert back to string
        contactMaskFiles = [os.path.splitext(x)[0] for x in contactMaskFiles]
        contactMaskFiles = list(map(int, contactMaskFiles))
        contactMaskFiles.sort()
        contactMaskFiles = [str(x) + '.npy' for x in contactMaskFiles]

        matStructure['est_contactmasks'] = extractImages(contact_folder, contactMaskFiles)

    filename = "gelsight_data" + '.mat'
    savePath = os.path.join(base_folder, filename)
    if matStructure:
        scipy.io.savemat(savePath, matStructure)

    print(savePath)

    if os.path.exists(fcrn_folder):
        print('Saving FCRN heightmap')
        matStructure = {}
        genHeightmapPath = fcrn_folder
        genHeightmapFiles = (os.listdir(genHeightmapPath))
        genHeightmapFiles = [x for x in genHeightmapFiles if any(c.isdigit() for c in x)]

        # strip file names, sort numeric, and convert back to string
        genHeightmapFiles = [os.path.splitext(x)[0] for x in genHeightmapFiles]
        genHeightmapFiles = list(map(int, genHeightmapFiles))
        genHeightmapFiles.sort()
        genHeightmapFiles = [str(x) + '.npy' for x in genHeightmapFiles]

        matStructure['fcrn_heightmaps'] = extractImages(genHeightmapPath, genHeightmapFiles)
        matStructure['fcrn_normalmaps'] = extractNormals(genHeightmapPath, genHeightmapFiles)
        
        filename = "fcrn_data" + '.mat'
        savePath = os.path.join(fcrn_folder, filename)
        if matStructure:
            scipy.io.savemat(savePath, matStructure)
        print(savePath)

    if os.path.exists(mlp_folder):
        print('Saving MLP heightmap')
        matStructure = {}
        genHeightmapPath = mlp_folder
        genHeightmapFiles = (os.listdir(genHeightmapPath))
        genHeightmapFiles = [x for x in genHeightmapFiles if any(c.isdigit() for c in x)]

        genHeightmapFiles = [os.path.splitext(x)[0] for x in genHeightmapFiles]
        genHeightmapFiles = list(map(int, genHeightmapFiles))
        genHeightmapFiles.sort()
        genHeightmapFiles = [str(x) + '.npy' for x in genHeightmapFiles]

        matStructure['mlp_heightmaps'] = extractImages(genHeightmapPath, genHeightmapFiles)
        matStructure['mlp_normalmaps'] = extractNormals(genHeightmapPath, genHeightmapFiles)

        filename = "mlp_data" + '.mat'
        savePath = os.path.join(mlp_folder, filename)
        if matStructure:
            scipy.io.savemat(savePath, matStructure)
        print(savePath)

    if os.path.exists(lookup_folder):
        print('Saving Lookup heightmap')
        matStructure = {}
        genHeightmapPath = lookup_folder
        genHeightmapFiles = (os.listdir(genHeightmapPath))
        genHeightmapFiles = [x for x in genHeightmapFiles if any(c.isdigit() for c in x)]

        genHeightmapFiles = [os.path.splitext(x)[0] for x in genHeightmapFiles]
        genHeightmapFiles = list(map(int, genHeightmapFiles))
        genHeightmapFiles.sort()
        genHeightmapFiles = [str(x) + '.npy' for x in genHeightmapFiles]

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

    print('\nreal2mat.py\n')
    # change path to cwd
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
    obj = str(sys.argv[1])
    dir = str(sys.argv[2])
    print(obj)
    base_folder = os.path.join(dir, 'ycbSight', obj)
    gelsight_folder = os.path.join(dir, 'ycbSight', obj, 'gelsight')

    fcrn_folder = os.path.join(dir, 'generated_data', 'fcrn', obj)
    mlp_folder = os.path.join(dir, 'generated_data', 'mlp', obj)
    lookup_folder = os.path.join(dir, 'generated_data', 'lookup', obj)
    contact_folder = os.path.join(dir, 'generated_data', 'contact_masks', obj)

    npy_to_matlab(base_folder, gelsight_folder, fcrn_folder, mlp_folder, lookup_folder, contact_folder)
