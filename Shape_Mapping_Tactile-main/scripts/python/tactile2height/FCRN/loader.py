import os
from os import path as osp
import numpy as np
import h5py
from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import sys
sys.path.append("..")
import FCRN.flow_transforms


class GelDataLoader(data.Dataset):
    def __init__(self, data_path, label_path):
        self.data_path = data_path # to txt file
        self.label_path = label_path # to txt file

        self.data = open(osp.join(data_path),'r')
        self.label = open(osp.join(label_path),'r')

        self.data_content = self.data.read()
        self.data_list = self.data_content.split("\n")

        self.label_content = self.label.read()
        self.label_list = self.label_content.split("\n")

    def __getitem__(self, index):
        img_path = self.data_list[index].split(" ")[1]
        dpt_path = self.label_list[index].split(" ")[1]
        image = None
        depth = None
        with Image.open(img_path) as im:
            # im.show()
            image = np.asarray(im)
        depth = np.load(dpt_path,allow_pickle=True)

        input_transform = transforms.Compose([flow_transforms.Scale(240),
                                              flow_transforms.ArrayToTensor()])
        target_depth_transform = transforms.Compose([flow_transforms.Scale_Single(240),
                                                     flow_transforms.ArrayToTensor()])

        img = input_transform(image)
        dpt = target_depth_transform(depth)

        return img, dpt

    def __len__(self):
        return len(self.label_list)-1

class TestDataLoader(data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        image = self.data
        input_transform = transforms.Compose([FCRN.flow_transforms.Scale(240),
                                              FCRN.flow_transforms.ArrayToTensor()])

        img = input_transform(image)
        return img

    def __len__(self):
        return 1
