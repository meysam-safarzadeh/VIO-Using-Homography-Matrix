#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 13:15:50 2022

@author: Meysam
"""
import numpy as np
import matplotlib.pyplot as plt


def trajectory_plotter(y_pred, y_true):
    
    inidces = [0, 2760, 3860, 5060]
    plot_name = [' Seq 05', ' Seq 07', ' Seq 10']
    
    
    
    for j in range(3):
        start = inidces[j]
        stop = inidces[j+1]
    
        # predict = regr_multirf.predict(X_train)
        # gt = y_train
        # print(start, stop)
    
        predict = y_pred[start:stop,:]
        gt = y_true[start:stop,:]
    
        x_predict = np.empty([predict.shape[0], 1], dtype=float)
        y_predict = np.empty([predict.shape[0], 1], dtype=float)
    
        x_gt = np.empty([gt.shape[0], 1], dtype=float)
        y_gt = np.empty([gt.shape[0], 1], dtype=float)
    
        a = 0.00
        b = 0.00
        c = 0.00
        d = 0.00
    
        for i in range(predict.shape[0]):
            a = a + predict[i, 0]
            x_predict[i] = np.round(a, decimals=4)
            
            b = b + predict[i, 2]
            y_predict[i] = np.round(b, decimals=4)
            
            c = c + gt[i, 0]
            x_gt[i] = np.round(c, decimals=4)
            
            d = d + gt[i, 2]
            y_gt[i] = np.round(d, decimals=4)
            #print(b)
            
        plt.figure(dpi=300)
        plt.scatter(x_predict, y_predict,  color='g', label='predict', s=1)
        plt.scatter(x_gt, y_gt,  color='r', label='GT', s=1)
        plt.legend()
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.title('Trajectory' )
        plt.show()


