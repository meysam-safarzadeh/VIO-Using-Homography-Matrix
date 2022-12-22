#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 14:42:57 2022

@author: Meysam
this script outputs a csv file consists of imu data and coresponding timestamps.
"""

import numpy as np
import os
# import matplotlib
import datetime
import pandas as pd


# In[] loading imu list
folder_num = 9 # you should change this folder number variable
sample_num = [270, 2760, 1100, 1100, 4070, 1590, 1200, 4540, 4660, 1100]
KITTI_dir = '/home/dulab/selective_feature_fusion/KITTI/'
seq_folder = os.listdir(KITTI_dir)
seq_folder.sort()
imu_dir =  os.path.join(KITTI_dir, seq_folder[folder_num], 'oxts', 'data', '')

imu_list = sorted(os.listdir(imu_dir))
imu_list = [os.path.join(imu_dir, i) for i in imu_list]

imu_value = []
for i in range(len(imu_list)):
    imu_i = np.loadtxt(imu_list[i])
    imu_i = np.concatenate((imu_i[11:14], imu_i[17:20]))
    imu_value.append(imu_i)

imu_value = np.array(imu_value)

print('imu shape: ' ,imu_value.shape)

# In[] sample IMU with network input shape

timestampstxt = pd.read_csv(os.path.join(KITTI_dir, seq_folder[folder_num], 'oxts', 'timestamps.txt'),
                            delim_whitespace=True)
timestampstxt = np.array(timestampstxt)
timestamp = []

for i in range(len(timestampstxt)):
    a= timestampstxt[i,1]
    a_time = datetime.datetime.strptime(a[:-3], "%H:%M:%S.%f")
    a_timedelta = a_time - datetime.datetime(1900,1,1)
    time_stamp = a_timedelta.total_seconds() # in sec
    timestamp.append(time_stamp)

timestamp = np.array(timestamp)

print('timestamp shape: ', timestamp.shape)

# In[] save the imu list to oxts with new format


output = np.column_stack((timestamp, imu_value))

final_output = pd.DataFrame(output, columns=['timestamps (sec)', 'acc_x (m/s^2)', 'acc_y (m/s^2)', 
                              'acc_z (m/s^2)', 'omega_x (rad/s)',
                              'omega_y (rad/s)', 'omega_z (rad/s)'])

final_output.to_csv(os.path.join(KITTI_dir, seq_folder[folder_num], 'oxts/', 'imu.csv'))

