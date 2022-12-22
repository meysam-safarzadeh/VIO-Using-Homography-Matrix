#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 21:05:22 2022

@author: Meysam
this file is written to convert the image timestamps in txt format to timestamps in 
csv format.
"""


import pandas as pd
import os
import datetime
import numpy as np


# In[] load the timestamps from the txt file

folder_num = 10 # you should change this folder number variable
KITTI_dir = '/home/dulab/selective_feature_fusion/KITTI_sync/'
seq_folder = os.listdir(KITTI_dir)
seq_folder.sort()

timestamp_dir =  os.path.join(KITTI_dir, seq_folder[folder_num], 'image_02', 'timestamps.txt')
timestampstxt = pd.read_csv(timestamp_dir, delim_whitespace=True)

timestampstxt = np.array(timestampstxt)
timestamp = []

for i in range(len(timestampstxt)):
    a= timestampstxt[i,1]
    a_time = datetime.datetime.strptime(a[:-3], "%H:%M:%S.%f")
    a_timedelta = a_time - datetime.datetime(1900,1,1)
    time_stamp = a_timedelta.total_seconds() # in sec
    timestamp.append(time_stamp)

output = np.array(timestamp)


# In[] save the timestamps in a csv file



final_output = pd.DataFrame(output, columns=['timestamps (sec)'])
final_output.to_csv(os.path.join(KITTI_dir, seq_folder[folder_num], 'image_02/', 'timestamps.csv'))





