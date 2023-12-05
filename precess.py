import os
import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image


data_dir = 'art_img/'
save_dir = 'art_img_32/'
# class_name = ['airplane', 'bee', 'bicycle', 'bird', 'book', 'bus', 'chair'] # 
class_name = os.listdir(data_dir)


for class_idx in range(len(class_name)):
    print('class:',class_idx)
    label = class_idx
    file_list = os.listdir(data_dir + class_name[class_idx])
    n_file = len(file_list)
    print('total images:', n_file)
    for i in range(n_file):
        # train_image = plt.imread(data_dir + class_name[class_idx] + '/' + file_list[i]).transpose((2,0,1))
        im = Image.open(data_dir + class_name[class_idx] + '/' + file_list[i])
        im = im.resize((32,32))
        save_path = save_dir + class_name[class_idx] + '/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        im.save(save_path + file_list[i])