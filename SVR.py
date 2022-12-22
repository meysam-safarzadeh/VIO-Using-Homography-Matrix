#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 16:10:29 2022

@author: Meysam
"""

import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from trajectory_plotter import trajectory_plotter
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn import preprocessing

# load data
pose = np.array(pd.read_csv('./dataset_csv/pose.csv'))
pose = pose[:, 1:]
imu = np.array(pd.read_csv('./dataset_csv/imu.csv'))
imu = imu[:, 1:]
H = np.array(pd.read_csv('./dataset_csv/homography_matrixes.csv'))
H = H[:, 1:]



X = np.hstack((imu, H))
y = pose[:, :3]

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5)

# Set the parameters by cross-validation
# parameters = {'kernel': ('linear', 'rbf', 'poly'),
#               'C': [1.5, 10],
#               'gamma': [1e-7, 1e-4],
#               'epsilon': [0.1, 0.2, 0.5, 0.3]}
# svr = svm.SVR()
# clf = GridSearchCV(svr, parameters)
# clf.fit(X_train, y_train)
# print(clf.best_params_)
# SVR
regr = make_pipeline(StandardScaler(), MultiOutputRegressor(SVR(gamma='auto',  C=1.0, kernel='rbf')))
# regr = MultiOutputRegressor(SVR(gamma='auto',  C=1.0))
regr.fit(X_train, y_train)
print("resluts of SVR")
importances = regr.steps[1][1].estimators_
# print(importances[0].coef_)


mae_train = mean_absolute_error(y_train, regr.predict(X_train))
mae_test = mean_absolute_error(y_test, regr.predict(X_test))
print("MAE train: {:.4f}".format(mae_train))
print("MAE test: {:.4f}".format(mae_test))

#
#
# # plot trajectory
# trajectory_plotter(regr.predict(X), y)





X = imu
y = pose[:, :3]

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5)

# Set the parameters by cross-validation
# parameters = {'kernel': ('linear', 'rbf', 'poly'),
#               'C': [1.5, 10],
#               'gamma': [1e-7, 1e-4],
#               'epsilon': [0.1, 0.2, 0.5, 0.3]}
# svr = svm.SVR()
# clf = GridSearchCV(svr, parameters)
# clf.fit(X_train, y_train)
# print(clf.best_params_)
# SVR
regr = make_pipeline(StandardScaler(), MultiOutputRegressor(SVR(gamma='auto',  C=1.0, kernel='rbf')))
# regr = MultiOutputRegressor(SVR(gamma='auto',  C=1.0))
regr.fit(X_train, y_train)
print("resluts of SVR (inertial only)")
importances = regr.steps[1][1].estimators_
# print(importances[0].coef_)


mae_train = mean_absolute_error(y_train, regr.predict(X_train))
mae_test = mean_absolute_error(y_test, regr.predict(X_test))
print("MAE train: {:.4f}".format(mae_train))
print("MAE test: {:.4f}".format(mae_test))

#
#
# # plot trajectory
# trajectory_plotter(regr.predict(X), y)











X = H
y = pose[:, :3]

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5)

# Set the parameters by cross-validation
# parameters = {'kernel': ('linear', 'rbf', 'poly'),
#               'C': [1.5, 10],
#               'gamma': [1e-7, 1e-4],
#               'epsilon': [0.1, 0.2, 0.5, 0.3]}
# svr = svm.SVR()
# clf = GridSearchCV(svr, parameters)
# clf.fit(X_train, y_train)
# print(clf.best_params_)
# SVR
regr = make_pipeline(StandardScaler(), MultiOutputRegressor(SVR(gamma='auto',  C=1.0, kernel='rbf')))
# regr = MultiOutputRegressor(SVR(gamma='auto',  C=1.0))
regr.fit(X_train, y_train)
print("resluts of SVR (vision only)")
importances = regr.steps[1][1].estimators_
# print(importances[0].coef_)


mae_train = mean_absolute_error(y_train, regr.predict(X_train))
mae_test = mean_absolute_error(y_test, regr.predict(X_test))
print("MAE train: {:.4f}".format(mae_train))
print("MAE test: {:.4f}".format(mae_test))

#
#
# # plot trajectory
# trajectory_plotter(regr.predict(X), y)





