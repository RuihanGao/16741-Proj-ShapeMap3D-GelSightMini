import os
from os import path as osp
import torch
import h5py
import numpy as np
import cv2
from PIL import Image
from FCRN.fcrn import FCRN_net
from torch.autograd import Variable
from FCRN.loader import TestDataLoader
# import matplotlib.pyplot as plot
import FCRN.flow_transforms
import torchvision.transforms as transforms
from config import fcrn_net_config, real_fcrn_net_config
# import matplotlib.pyplot as plt
from PIL import Image

class Model:
    def __init__(self, **config):
        # print("setting devices...")
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if use_cuda else "cpu")
        # print(self.device)

        # print("setting parameters...")
        self.batch_size = 1
        self.params = {'batch_size': self.batch_size,
                'shuffle': False}

        self.model = FCRN_net(self.batch_size)
        checkpoint = torch.load(config['path2model'],map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()

    def test(self, test_data):
        # test_data: tactile img 640 * 480
        # result: height map 640 * 480

        test_set = TestDataLoader(test_data)
        test_loader = torch.utils.data.DataLoader(test_set, **self.params)
        with torch.no_grad():
            for data in test_loader:
                data = data.to(self.device)
                output = self.model(data.float())[0].data.squeeze().numpy().astype(np.float32)
                return output

class Simulation:
    def __init__(self,**config):
        self.data_folder = config['path2data']
        self.save_folder = config['path2save']
        self.num_data = config['num_data']
        self.image2heightmap = Model(**config)

    def simulate(self, object=None):
        if object is None:
            # generate height maps for all objects
            object_folders = sorted(os.listdir(self.data_folder))
        else:
            object_folders = [object]

        #### DEBUG ####
        # img_path = "/media/suddhu/Backup Plus/suddhu/rpl/datasets/tactile_mapping/calib_real/calib_ball/gelsight/x.jpg"
        # with Image.open(img_path) as im:
        #     # im.show()
        #     img = np.asarray(im)

        # height_map = self.image2heightmap.test(img)
        # height_map = cv2.resize(height_map, dsize=(640, 480), interpolation=cv2.INTER_CUBIC)
        # np.save("/media/suddhu/Backup Plus/suddhu/rpl/datasets/tactile_mapping/calib_real/real_gelmap2.npy", height_map)

        # plt.figure(1)
        # plt.subplot(111)
        # plt.imshow(height_map)

        # plt.show(block=False)
        # p = input("pause")
        #### DEBUG ####

        for obj in object_folders:
            if obj == ".DS_Store":
                continue
            print(obj)
            data_folder = self.data_folder + obj + '/tactile_imgs/'
            if not os.path.exists(data_folder):
                data_folder = self.data_folder + obj + '/'

            save_folder = self.save_folder + obj + '/'
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)

            tactileImageFiles = sorted(os.listdir(data_folder), key=lambda y: int(y.split("_")[1]))

            for idx in range(self.num_data):
                # img = cv2.imread(data_folder + str(idx) + ‘.jpg’)
                # img = img.astype(int)

                img_path = data_folder + str(idx) + ".jpg"
                if not os.path.exists(img_path):
                    img_path = os.path.join(data_folder, tactileImageFiles[idx])

                with Image.open(img_path) as im:
                    # im.show()
                    img = np.asarray(im)

                height_map = self.image2heightmap.test(img)
                # print(height_map.shape)
                # print(np.max(height_map))
                # print(np.min(height_map))

                # fig = plt.figure(1)
                # ax1 = fig.add_subplot(221)
                # ax1.title.set_text('Image')
                # plt.imshow(img)

                # ax2 = fig.add_subplot(222)
                # ax2.title.set_text('FCRN heightmap')
                # plt.imshow(height_map)

                # plt.show()

                np.save(save_folder + str(idx) + '.npy', height_map)


if __name__ == "__main__":
    # sim = Simulation(**fcrn_net_config)
    sim = Simulation(**real_fcrn_net_config)
    obj = '036_wood_block'
    sim.simulate(obj)
    # sim.simulate()
