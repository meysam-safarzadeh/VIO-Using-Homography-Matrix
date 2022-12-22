#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 17:28:50 2022

@author: Meysam
"""

import pandas as pd
import numpy as np
import time
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from trajectory_plotter import trajectory_plotter
import matplotlib.pyplot as plt



# load data
pose = np.array(pd.read_csv('./dataset_csv/pose.csv'))
pose = pose[:,1:]
# print(pose.shape)
imu = np.array(pd.read_csv('./dataset_csv/imu.csv'))
imu = imu[:,1:]
# print(imu.shape)
H = np.array(pd.read_csv('./dataset_csv/homography_matrixes.csv'))
H = H[:,1:]
# print(H.shape)

print("Results of RFR")
X = np.hstack((imu, H))
y = pose[:,:3]

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# grid search


# RFR
max_depth = 30
regr_multirf = MultiOutputRegressor(
    RandomForestRegressor(n_estimators=300, max_depth=max_depth, random_state=0, n_jobs=-1))
regr_multirf.fit(X_train, y_train)


importances = regr_multirf.estimators_[0].feature_importances_

start_time = time.time()
std = np.std([tree.feature_importances_ for tree in regr_multirf.estimators_[0]], axis=0)
elapsed_time = time.time() - start_time

forest_importances = pd.Series(importances)

fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()
# print(importances.shape)
sum_vision = np.sum(importances[60: 68])
# print(sum_vision)
sum_imu = np.sum(importances[0: 59])
# print(sum_imu)

mae_train = mean_absolute_error(y_train, regr_multirf.predict(X_train))
mae_test = mean_absolute_error(y_test, regr_multirf.predict(X_test))
print("MAE train: {:.4f}".format(mae_train))
print("MAE test: {:.4f}".format(mae_test))

# trajectory_plotter(regr_multirf.predict(X), y)














print("Results of RFR (inertial only)")
X = imu
y = pose[:,:3]

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# grid search


# RFR
max_depth = 30
regr_multirf = MultiOutputRegressor(
    RandomForestRegressor(n_estimators=300, max_depth=max_depth, random_state=0, n_jobs=-1))
regr_multirf.fit(X_train, y_train)


importances = regr_multirf.estimators_[0].feature_importances_

start_time = time.time()
std = np.std([tree.feature_importances_ for tree in regr_multirf.estimators_[0]], axis=0)
elapsed_time = time.time() - start_time

forest_importances = pd.Series(importances)

fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()
# print(importances.shape)
sum_vision = np.sum(importances[60: 68])
# print(sum_vision)
sum_imu = np.sum(importances[0: 59])
# print(sum_imu)

mae_train = mean_absolute_error(y_train, regr_multirf.predict(X_train))
mae_test = mean_absolute_error(y_test, regr_multirf.predict(X_test))
print("MAE train: {:.4f}".format(mae_train))
print("MAE test: {:.4f}".format(mae_test))

# trajectory_plotter(regr_multirf.predict(X), y)







print("Results of RFR (vision only)")
X = H
y = pose[:,:3]

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# grid search


# RFR
max_depth = 30
regr_multirf = MultiOutputRegressor(
    RandomForestRegressor(n_estimators=300, max_depth=max_depth, random_state=0, n_jobs=-1))
regr_multirf.fit(X_train, y_train)


importances = regr_multirf.estimators_[0].feature_importances_

start_time = time.time()
std = np.std([tree.feature_importances_ for tree in regr_multirf.estimators_[0]], axis=0)
elapsed_time = time.time() - start_time

forest_importances = pd.Series(importances)

fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()
# print(importances.shape)
sum_vision = np.sum(importances[60: 68])
# print(sum_vision)
sum_imu = np.sum(importances[0: 59])
# print(sum_imu)

mae_train = mean_absolute_error(y_train, regr_multirf.predict(X_train))
mae_test = mean_absolute_error(y_test, regr_multirf.predict(X_test))
print("MAE train: {:.4f}".format(mae_train))
print("MAE test: {:.4f}".format(mae_test))

# trajectory_plotter(regr_multirf.predict(X), y)






