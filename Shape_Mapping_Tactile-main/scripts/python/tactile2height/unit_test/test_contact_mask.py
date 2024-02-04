import matplotlib.pyplot as plt
import numpy as np
import os
from os import path as osp
import cv2
import scipy
from skimage.color import rgb2hsv


def simulate(data_folder, mask_folder, img_folder, object=None):
    if object is None:
        # generate height maps for all objects
        object_folders = sorted(os.listdir(data_folder))
    else:
        object_folders = [object]

    for obj in object_folders:
        if obj == ".DS_Store":
            continue
        print(obj)
        cur_folder = data_folder + obj + '/'
        tactile_folder = img_folder + obj + '/'
        contact_mask_folder = mask_folder + obj + '/'
        heightMaps = sorted(os.listdir(cur_folder), key=lambda y: int(y.split(".")[0]))
        tactiles = sorted(os.listdir(tactile_folder), key=lambda y: int(y.split("_")[1]))
        contacts = sorted(os.listdir(contact_mask_folder), key=lambda y: int(y.split(".")[0]))

        for i, heightMap in enumerate(heightMaps):
            if i % 2 == 0:
                continue
            print(heightMap)
            # img = cv2.imread(data_folder + str(idx) + ‘.jpg’)
            # img = img.astype(int)

            img_path = cur_folder + heightMaps[i]
            last_img_path = cur_folder + heightMaps[i-1]
            tactile = tactile_folder + tactiles[i]
            last_tactile = tactile_folder + tactiles[i-1]
            contact_path = contact_mask_folder + contacts[i]

            height_map = np.load(img_path)
            height_bg = np.load(last_img_path)
            height_diff = np.abs(height_map-height_bg)
            tactile_img = cv2.imread(tactile)
            last_tactile_img = cv2.imread(last_tactile)
            cur_contact = np.load(contact_path)

            cur_contact = 1*(cur_contact > 0)
            (cnts, _) = cv2.findContours(cur_contact.astype(np.uint8), #Input Image
                              cv2.RETR_EXTERNAL,           #Contour Retrieval Method
                              cv2.CHAIN_APPROX_SIMPLE)     #Contour Approximation Method

            contours = cnts
            print("Num contours: " + str(len(contours)))
            # cv2.drawContours(img, contours, -1, (0,255,0), 3)

            valid_contours = []
            for contour in contours:
                AreaCount=cv2.contourArea(contour)
                print(AreaCount)
                if AreaCount<1000:
                    continue
                valid_contours.append(contour)


            # mask = np.zeros_like(img) # Create mask where white is what we want, black otherwise
            mask = np.zeros((cur_contact.shape[0],cur_contact.shape[1]))
            cv2.drawContours(mask, valid_contours, -1, 1, -1) # Draw filled contour in mask
            out = np.zeros_like(cur_contact) # Extract out the object and place into output image
            out[mask == 1] = cur_contact[mask == 1]

            diff_img = np.abs(np.sum(tactile_img.astype(float) - last_tactile_img.astype(float),axis=2))
            contact_mask = 1.0*(diff_img > np.percentile(diff_img, 90)*0.5) #*0.8
            print(np.min(contact_mask))
            cv2.imshow('contact mask', contact_mask.astype('uint8'))
            cv2.waitKey(0)
            dim = (320, 240)
            contact_mask = cv2.resize(contact_mask.astype('uint8'), dim, interpolation = cv2.INTER_AREA)
            mask = (mask.astype('uint8') & contact_mask.astype('uint8'))
            print(np.max(mask.astype('uint8')))

            # Show the output image


            cv2.imshow('mask', mask)
            cv2.waitKey(0)

            cv2.imshow('heightmap', height_map/100.0)
            cv2.waitKey(0)
            # cv2.destroyAllWindows()

            # cur_hsv = rgb2hsv(tactile_img)
            # last_hsv = rgb2hsv(last_tactile_img)
            #
            # plt.figure(1)
            # plt.subplot(231)
            # fig = plt.imshow(cur_hsv[:,:,0])
            #
            # plt.subplot(232)
            # fig = plt.imshow(cur_hsv[:,:,1])
            #
            # plt.subplot(233)
            # fig = plt.imshow(cur_hsv[:,:,2])
            #
            # plt.subplot(234)
            # fig = plt.imshow(last_hsv[:,:,0])
            #
            # plt.subplot(235)
            # fig = plt.imshow(last_hsv[:,:,1])
            #
            # plt.subplot(236)
            # fig = plt.imshow(last_hsv[:,:,2])
            # plt.show()
            # p = input("pause")


            # # # plt.imshow(height_map)
            # # # plt.show()
            # #
            # plt.figure(1)
            # plt.subplot(221)
            # fig = plt.imshow(height_map)
            #
            # plt.subplot(222)
            # fig = plt.imshow(tactile_img)
            #
            # plt.subplot(223)
            # fig = plt.imshow(contact_mask)
            #
            # plt.subplot(224)
            # fig = plt.imshow(cur_contact)
            # plt.show()
            # p = input("pause")



if __name__ == "__main__":
    # data_folder = '../../../../generated_data/real/fcrn/'
    # data_folder = '../../../../generated_data/textured_50sampled/contact_mask_fcrn/'
    data_folder = '../../../../generated_data/real/fcrn/'
    mask_folder = '../../../../generated_data/real/contact_mask_fcrn/'
    tactile_folder = '../../../../gelsight_data/real/'
    obj = '021_bleach_cleanser'
    simulate(data_folder, mask_folder, tactile_folder,object=obj)
