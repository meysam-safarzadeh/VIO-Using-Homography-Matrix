#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 19:16:11 2022

@author: Meysam
"""
import cv2
import numpy as np

def SIFT(img1, img2):
    sift = cv2.xfeatures2d.SIFT_create()

    kp1, descriptors_1 = sift.detectAndCompute(img1,None) # kp1 == keypoints_1
    kp2, descriptors_2 = sift.detectAndCompute(img2,None) # kp2 == keypoints_2
    
    return kp1, descriptors_1, kp2, descriptors_2


def ORB_feature_detection(img1, img2):
    orb = cv2.ORB_create()

    kp1, descriptors_1 = orb.detectAndCompute(img1,None) # kp1 == keypoints_1
    kp2, descriptors_2 = orb.detectAndCompute(img2,None) # kp2 == keypoints_2
    descriptors_1 = np.float32(descriptors_1)
    descriptors_2 = np.float32(descriptors_2)
    
    return kp1, descriptors_1, kp2, descriptors_2