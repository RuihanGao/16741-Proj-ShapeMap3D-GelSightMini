import numpy as np
import os
from os import path as osp
from PIL import Image

save_path = osp.join('..','..','..','dataset')
train_data_file = open(osp.join(save_path,'train_data.txt'),'r')

data_content = train_data_file.read()
data_list = data_content.split("\n")

print(len(data_list))
print(data_list[-1])
print(data_list[0].split(" ")[1])
img_path = data_list[0].split(" ")[1]
with Image.open(img_path) as im:
    # im.show()
    print(np.asarray(im).shape)
