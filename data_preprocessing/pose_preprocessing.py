#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 11:20:04 2022

@author: Meysam
"""
import numpy as np
from scipy.spatial.transform import Rotation as R
from from pyquaternion import Quaternion


# In[] load the data

# drag and drop the txt file of the ground truth
# then change d_xx name to what you named the data
seq_name = '/seq_10_'
data = np.array(d_10) 
trans = np.stack((data[:,3], data[:,7], data[:,11]), axis=1)
rot = []


for i in range(len(data)):
    rot_mat = np.stack((data[i,0:3], data[i,4:7], data[i,8:11]), axis=0)
    print(rot_mat.shape, '**********')
    rot.append(rot_mat)
    
rot = np.array(rot)

print(rot.shape, trans.shape)

# In[] Rotation to quaternion
relative_rot = []
path_to_save = '/home/dulab/KITTI_preprocessed/pose'

for i in range(len(rot)-1):
    r_i = R.from_matrix(rot[i,:,:])
    r_i_inv = r_i.inv()
    
    r_next_i = R.from_matrix(rot[i+1,:,:])
    
    print(quat_i, quat_i.shape)
    
    relative_quat = r_next_i * r_i_inv
    relative_euler = relative_quat.as_euler('xyz')
    
    print(relative_euler)
    relative_rot.append(relative_euler)
relative_rot = np.array(relative_rot) # euler angles
print(relative_rot.shape)



# In[] save

path_to_save = '/home/dulab/KITTI_preprocessed/pose'
final_pose = []
for i in range(len(relative_rot)):
    relative_trans = trans[i+1,:] - trans[i,:]
    pose = np.concatenate((relative_trans, relative_rot[i,:]), axis=0)
    print(pose, pose.shape)
    file = path_to_save + seq_name + f"{i:04}"
    np.save(file, pose)
    final_pose.append(pose)


final_pose = np.array(final_pose)

print(final_pose.shape)


