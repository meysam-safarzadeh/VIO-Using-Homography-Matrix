#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 11:20:08 2022

@author: Meysam
"""


import os
os.chdir('/source_code/homography_estimation')



import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from homography.feature_detection import ORB_feature_detection
from homography.feature_matcher import FLANN
from homography.utils import (SNN_ratio_test, 
                                 draw_matches)
from homography.outlier_removal import magsac



# In[] read images




data_dir = '/frame' # this should be the raw frame directory gathered from KITTI
os.chdir(data_dir)
img_list = sorted(os.listdir(data_dir))
num_inliers = np.empty((len(img_list),))
H_matrixes = []

save = False

i = img_list[5]

for i in range(len(img_list)-1):
    # i=2
    im_src_dir = img_list[i]
    im_dst_dir = img_list[i+1]

    
    img_src_0 = cv2.imread(im_src_dir)
    img_dst_0 = cv2.imread(im_dst_dir)
    
    img1 = cv2.cvtColor(img_src_0, cv2.COLOR_RGB2GRAY)
    img2 = cv2.cvtColor(img_dst_0, cv2.COLOR_RGB2GRAY)
    
    figure, ax = plt.subplots(1, 2, figsize=(16, 8))
    
    ax[0].imshow(img1, cmap='gray')
    ax[1].imshow(img2, cmap='gray')
    
    
    ## In[] Homography estimation
    
    kps1, descriptors_1, kps2, descriptors_2 = ORB_feature_detection(img1, img2)
    matches = FLANN(descriptors_1, descriptors_2)
    
    tentatives = SNN_ratio_test(matches, SNN_threshold=0.7)
    draw_matches(kps1, kps2, tentatives, img1, img2)
    H, mask, num_inliers[i] = magsac(kps1, kps2, tentatives,
                                      10,
                                      True,
                                      sigma_th=10)
    H_matrixes.append(H)
    
    if num_inliers[i]<5:
        continue
    print('\n --------------------- number:', i)
    draw_matches(kps1, kps2, tentatives, img1, img2, H, mask)
    
 
   


H_matrix = np.array(H_matrixes)
new_H = np.reshape(H_matrix, (22390,9))
pd.DataFrame(new_H).to_csv('/source_code/dataset_csv/homography_matrixes.csv', sep=',')
