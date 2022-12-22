#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 17:10:28 2022

@author: Meysam
this file aims to sample frames to feed them to the network.
They should be stacked together and have the shape (h,w,6)
"""


import numpy as np
import os
from PIL import Image
from matplotlib.pyplot import imshow


# In[]

folder_num = 9 # you should change this folder number variable
seq_num = [4,5,6,7,8,9,10,0,2,1]
seq_name = '/seq_' + f"{seq_num[folder_num]:02}" + '_'
KITTI_sync_dir = '/home/dulab/selective_feature_fusion/KITTI_sync/'
path_to_save = '/home/dulab/KITTI_preprocessed/frame'

seq_sync_list = sorted(os.listdir(KITTI_sync_dir))
image_path = os.path.join(KITTI_sync_dir, seq_sync_list[folder_num+2],
                          'image_02',
                          'data')

entries = sorted(os.listdir(image_path))



# In[]


for i in range(round(len(entries)-1)):
    file_name_1 = os.path.join(image_path, entries[i])
    file_name_2 = os.path.join(image_path, entries[i+1])
    
    # print(file_name_1)
    # print(file_name_2)
    
    im_1 = Image.open(file_name_1)
    im_2 = Image.open(file_name_2)
    
    im_1 = np.array(im_1)
    im_2 = np.array(im_2)
    
    
    # stack for gray images or concatenate for RGB images
    im = np.concatenate((im_1, im_2), axis=2) 
    
    
    file = path_to_save + seq_name + f"{i:04}"
    np.save(file, im)
    print(i)
    
    