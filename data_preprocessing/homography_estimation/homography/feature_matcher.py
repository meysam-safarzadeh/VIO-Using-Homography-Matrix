#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 19:15:16 2022

@author: Meysam
"""
import cv2


def FLANN(descriptors_1, descriptors_2, 
          FLANN_INDEX_KDTREE=1, 
          trees=5, 
          checks = 50):
    
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees=trees)
    search_params = dict(checks=checks)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors_1, descriptors_2, k=2)
    
    return matches




def Brute_force(descriptors_1, descriptors_2, k=2):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors_1, descriptors_2, k)
    
    return matches