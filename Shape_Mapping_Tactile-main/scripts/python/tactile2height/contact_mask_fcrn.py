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
from config import contact_mask_fcrn_config, real_contact_mask_fcrn_config
import matplotlib.pyplot as plt
from PIL import Image

def extractContour(cur_img,init_img):
    # MarkerThresh=-30
    MarkerThresh = -5
    diff = cur_img-init_img
    max_img = np.amax(diff,2)
    MarkerMask = max_img<MarkerThresh
    # cv2.imshow('mask', MarkerMask.astype(np.uint8))
    # cv2.waitKey(0)
    areaThresh1=50
    areaThresh2=400
    MarkerCenter=np.empty([0, 3])
    # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # des = cv2.bitwise_not(gray)
    # cnts,hier = cv2.findContours(des,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    (cnts, _) = cv2.findContours(MarkerMask.astype(np.uint8), #Input Image
                              cv2.RETR_EXTERNAL,           #Contour Retrieval Method
                              cv2.CHAIN_APPROX_SIMPLE)     #Contour Approximation Method

    contours = cnts
    print("Num contours: " + str(len(contours)))
    # cv2.drawContours(img, contours, -1, (0,255,0), 3)

    # mask = np.zeros_like(img) # Create mask where white is what we want, black otherwise
    mask = np.zeros((img.shape[0],img.shape[1]))
    cv2.drawContours(mask, contours, -1, 255, -1) # Draw filled contour in mask
    out = np.zeros_like(img) # Extract out the object and place into output image
    out[mask == 255] = img[mask == 255]

    # Show the output image
    # cv2.imshow('Output', mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return mask

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
                if idx % 2 == 0:
                    continue
                # img = cv2.imread(data_folder + str(idx) + ‘.jpg’)
                # img = img.astype(int)

                img_path = data_folder + str(idx) + ".jpg"
                if not os.path.exists(img_path):
                    img_path = os.path.join(data_folder, tactileImageFiles[idx])



                last_tactile_img = cv2.imread(os.path.join(data_folder, tactileImageFiles[idx-1]))
                tactile_img = cv2.imread(os.path.join(data_folder, tactileImageFiles[idx]))
                diff_img = np.abs(np.sum(tactile_img.astype(float) - last_tactile_img.astype(float),axis=2))
                contact_mask = diff_img > np.percentile(diff_img, 90)*0.8 #*0.8

                

                with Image.open(img_path) as im:
                    # im.show()
                    img = np.asarray(im)


                mask_map = self.image2heightmap.test(img)
                # print(height_map.shape)
                # print(np.max(height_map))
                # print(np.min(height_map))

                fig = plt.figure(1)
                ax1 = fig.add_subplot(221)
                ax1.title.set_text('Image')
                plt.imshow(tactile_img)

                ax2 = fig.add_subplot(222)
                ax2.title.set_text('FCRN contact map')
                plt.imshow(mask_map)

                ax3 = fig.add_subplot(223)
                ax3.title.set_text('threshold contact map')
                plt.imshow(contact_mask)

                plt.show()
                p = input("pause")

                # np.save(save_folder + str(idx) + '.npy', mask_map)


if __name__ == "__main__":
    # sim = Simulation(**fcrn_net_config)
    sim = Simulation(**real_contact_mask_fcrn_config)
    obj = '021_bleach_cleanser'
    sim.simulate(obj)
    # sim.simulate()
