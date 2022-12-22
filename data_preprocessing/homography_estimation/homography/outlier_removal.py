"""
Created on Wed Mar  2 16:40:40 2022

@author: Meysam
"""
import pymagsac
# import os
import cv2 
# import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy


def magsac(kps1, kps2, tentatives, 
           MIN_MATCH_COUNT = 10, 
           use_magsac_plus_plus=False, 
           sigma_th=10):
     
    if len(tentatives)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kps1[m.queryIdx].pt for m in tentatives ]).reshape(-1,2)
        dst_pts = np.float32([ kps2[m.trainIdx].pt for m in tentatives ]).reshape(-1,2)
        H, mask = pymagsac.findHomography(src_pts, dst_pts, use_magsac_plus_plus, sigma_th)
        
        if use_magsac_plus_plus:
            method = "MAGSAC++"
        else:
            method = "MAGSAC"
        num_inlier = deepcopy(mask).astype(np.float32).sum()
        print ('\n', num_inlier, 'inliers found using',  method)
    else:
        print( "Not enough matches are found - {}/{}".format(len(tentatives), MIN_MATCH_COUNT) )
        return None, None, 0
    return H, mask, num_inlier





def ransac(kps1, kps2, tentatives, 
           MIN_MATCH_COUNT = 10, 
           ransacReprojThreshold=3,
           maxIters=3000,
           confidence=0.99):
    
    if len(tentatives)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kps1[m.queryIdx].pt for m in tentatives ]).reshape(-1,1,2)
        dst_pts = np.float32([ kps2[m.trainIdx].pt for m in tentatives ]).reshape(-1,1,2)
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 
                                     ransacReprojThreshold,
                                     maxIters=maxIters,
                                     confidence=confidence)
        num_inlier = deepcopy(mask).astype(np.float32).sum()
        print ('\n', num_inlier, 'inliers found using RANSAC')
        return H, mask, num_inlier
    else:
        print( "Not enough matches are found - {}/{}".format(len(tentatives), MIN_MATCH_COUNT) )
        return
