#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 19:33:18 2022

@author: Meysam
"""

import numpy as np
import cv2
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os

def SNN_ratio_test(matches, SNN_threshold = 0.7):
    """
    # store all the good matches as per Lowe's ratio test. (SNN ratio test)
    """
    matches_all = [m for m,n in matches]
    matches_all = sorted(matches_all, key = lambda x:x.distance)

    
    good = [] # good == tentatives
    for m,n in matches:
        if m.distance < SNN_threshold*n.distance:
            good.append(m)
            
    return good


# def decolorize(img):
#     a = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
#     return a



def draw_matches(kps1, kps2, tentatives, img1, img2, 
                 H=None, mask=None, H_gt=None):
    if H is None:
        img3 = cv2.drawMatches(img1, kps1, img2, kps2, 
                               tentatives[:100], img2, flags=2)
        plt.figure()
        plt.title("Top 100 matches")
        plt.imshow(img3)
        plt.show()
        return
    matchesMask = mask.ravel().tolist()
    h,w = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts, H)
    
    
    
    img2_tr = cv2.polylines(img2,[np.int32(dst)],
                            True,(255,0,0),3, cv2.LINE_AA)
    
    if H_gt != None:
        # Ground truth transformation
        dst_GT = cv2.perspectiveTransform(pts, H_gt)
        img2_tr = cv2.polylines(deepcopy(img2_tr),[np.int32(dst_GT)],
                                True,(0,255,0),3, cv2.LINE_AA)
    
    
    # Blue is estimated, green is ground truth homography
    draw_params = dict(matchColor = (255,255,0), # draw matches in yellow color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)
    img_out = cv2.drawMatches(img1,kps1,img2_tr,kps2,tentatives,
                              None,**draw_params)
    plt.figure(figsize = (12,8))
    plt.title('Detected object in the scene')
    plt.imshow(img_out)
    return



def apply_persp_trans(img_src, img_dst, H, 
                      ROI_area=None, 
                      whitebalanced=None,
                      percentile=95):
    """
    Inputs:
        whitebalanced: if you need to do patch whitebalance, fill this with
        a list like: [from_row, from_column, row_width, column_width]
    Outputs:
        ROI image (uint8 RGB)
        Transformed image (uint8 RGB)
    """
    im_transformed = cv2.warpPerspective(img_dst, 
                                         np.linalg.pinv(H),
                                         (img_src.shape[1],img_src.shape[0]))
    im_transformed = cv2.cvtColor(im_transformed, cv2.COLOR_BGR2RGB)
    
    if whitebalanced:
            im_transformed = whitepatch_balancing(im_transformed, 
                                                  whitebalanced[0],
                                                  whitebalanced[1],
                                                  whitebalanced[2],
                                                  whitebalanced[3],
                                                  percentile=95)
            im_transformed = 255*im_transformed
            im_transformed = im_transformed.astype(np.uint8)
    plt.figure()
    plt.title('Transformed image')
    plt.imshow(im_transformed)
    plt.show()
    
    if ROI_area:
        ROI_img = im_transformed[ROI_area[0]:ROI_area[1], 
                                 ROI_area[2]:ROI_area[3]]
        plt.figure()
        plt.title("Detected ROI")
        plt.imshow(ROI_img)
        plt.show()
        return ROI_img, im_transformed
    return None, im_transformed



def whitepatch_balancing(image, from_row, from_column, 
                         row_width, column_width, percentile=95):
    
    fig, ax = plt.subplots(1,2, figsize=(10,5))
    ax[0].imshow(image)
    ax[0].add_patch(Rectangle((from_column, from_row), 
                              column_width, 
                              row_width, 
                              linewidth=3,
                              edgecolor='r', facecolor='none'));
    ax[0].set_title('Original transformed image')
    image_patch = image[from_row:from_row+row_width, 
                        from_column:from_column+column_width]
    image_max = (image*1.0 / 
                 np.percentile(image_patch, percentile, axis=(0,1))).clip(0, 1)
    # print(image_patch.max(axis=(0, 1)), np.percentile(image_patch, 70, axis=(0,1)))
    ax[1].imshow(image_max);
    ax[1].set_title('Whitebalanced Image')
    return image_max




def data_augmentation(arange_i, arange_j, img_src_0, img_dst_0, H, path, im_dst_dir):
    for i in np.arange(arange_i[0], arange_i[1], arange_i[2]):
        i = int(i)
        for j in np.arange(arange_j[0], arange_j[1], arange_j[2]):
            j = int(j)
            print(i,j)
            ROI_img, im_transformed = apply_persp_trans(img_src_0, 
                                                        img_dst_0, 
                                                        H, 
                                                        ROI_area=[80+i,170+i,
                                                                  100+j,175+j],
                                                        whitebalanced=False)
            plt.imsave(os.path.join(path, 
                                    im_dst_dir[:-4] + '_ROI_' + str(i) + 
                                    '_' + str(j) + '.jpg'),
                       ROI_img)

