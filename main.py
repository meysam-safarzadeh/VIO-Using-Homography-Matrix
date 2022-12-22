#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 5 00:19:25 2022

@author: Meysam
"""

import os

print("the results of MLP (early fusion):")
os.system('python early_fusion.py')



print("the results of MLP (inetermediate fusion):")
os.system('python intermediate_fusion.py')



print("the results of MLP (vision only):")
os.system('python early_fusion_visiononly.py')


print("the results of MLP (inertial only):")
os.system('python early_fusion_inertialonly.py')



print("the results of SVR:")
os.system('python SVR.py')




print("the results of RFR")
os.system('python RFR.py')




print("the results of GBR")
os.system('python GBR.py')
