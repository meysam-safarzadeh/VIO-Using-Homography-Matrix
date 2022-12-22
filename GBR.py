import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from trajectory_plotter import trajectory_plotter
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

# load data
pose = np.array(pd.read_csv('./dataset_csv/pose.csv'))
pose = pose[:, 1:]
imu = np.array(pd.read_csv('./dataset_csv/imu.csv'))
imu = imu[:, 1:]
H = np.array(pd.read_csv('./dataset_csv/homography_matrixes.csv'))
H = H[:, 1:]


print("Results of GBR")
X = np.hstack((imu, H))
y = pose[:, :3]

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5)

# model = MultiOutputRegressor(GradientBoostingRegressor(loss='absolute_error', learning_rate=0.1, n_estimators=100, subsample=1.0,
#                                                        criterion='friedman_mse', min_samples_split=2,
#                                                        min_samples_leaf=1,
#                                                        min_weight_fraction_leaf=0.0, max_depth=3,
#                                                        min_impurity_decrease=0.0, init=None, random_state=None,
#                                                        max_features=None,
#                                                        alpha=0.9, verbose=0, max_leaf_nodes=None, warm_start=False,
#                                                        validation_fraction=0.1, n_iter_no_change=None, tol=0.0001,
#                                                        ccp_alpha=0.0))
#
# hyperparameters = dict(estimator__learning_rate=[0.01, 0.05, 0.1, 0.15, 0.2], estimator__loss=['squared_error', 'absolute_error', 'huber', 'quantile'],
#                      estimator__n_estimators=[10, 20, 50, 100],
#                      estimator__criterion=['friedman_mse', 'squared_error'], estimator__min_samples_split=[2, 3, 5],
#                      estimator__max_depth=[15, 25, 30, 35, 45], estimator__min_samples_leaf=[2, 3, 5],
#                      estimator__min_impurity_decrease=[0, 0.2, 0.4, 0.6, 0.8, 1],
#                      estimator__max_leaf_nodes=[10, 20, 30, 50, 100])
#
# randomized_search = RandomizedSearchCV(model, hyperparameters, random_state=0, n_iter=5, scoring=None,
#                                        n_jobs=2, refit=True, cv=5, verbose=True,
#                                        pre_dispatch='2*n_jobs', error_score='raise', return_train_score=True)
#
# hyperparameters_tuning = randomized_search.fit(X_train, y_train)
# print('Best Parameters = {}'.format(hyperparameters_tuning.best_params_))
#
# tuned_model = hyperparameters_tuning.best_estimator_
tuned_model= MultiOutputRegressor(
                                GradientBoostingRegressor(loss='squared_error', learning_rate=0.15, n_estimators=20, subsample=1.0,
                                                       criterion='squared_error', min_samples_split=5, max_depth=15,
                                                       min_samples_leaf=3,
                                                       min_weight_fraction_leaf=0.0,
                                                       min_impurity_decrease=0.0, init=None, random_state=None,
                                                       max_features=None,
                                                       alpha=0.9, verbose=0, max_leaf_nodes=100, warm_start=False,
                                                       validation_fraction=0.1, n_iter_no_change=None, tol=0.0001,
                                                       ccp_alpha=0.0)
    )
tuned_model.fit(X_train, y_train)
mae_train = mean_absolute_error(y_train, tuned_model.predict(X_train))
mae_test = mean_absolute_error(y_test, tuned_model.predict(X_test))
print("MAE train: {:.4f}".format(mae_train))
print("MAE test: {:.4f}".format(mae_test))

# feature_importance = tuned_model.estimators_[0].feature_importances_
# print(feature_importance)
# sorted_idx = np.argsort(feature_importance)
# pos = np.arange(sorted_idx.shape[0]) + 0.5
# fig = plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)
# plt.barh(pos, feature_importance[sorted_idx], align="center")
# plt.yticks(pos, np.array(69)[sorted_idx])
# plt.title("Feature Importance (MDI)")
#
# result = permutation_importance(
#     tuned_model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2
# )
# sorted_idx = result.importances_mean.argsort()
# plt.subplot(1, 2, 2)
# plt.boxplot(
#     result.importances[sorted_idx].T,
#     vert=False,
#     labels=np.array(69)[sorted_idx],
# )
# plt.title("Permutation Importance (test set)")
# fig.tight_layout()
# plt.show()
# trajectory_plotter(tuned_model.predict(X), y)
# print(tuned_model.predict())











print("Results of GBR (inertial only):")
X = imu
y = pose[:, :3]

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5)

# model = MultiOutputRegressor(GradientBoostingRegressor(loss='absolute_error', learning_rate=0.1, n_estimators=100, subsample=1.0,
#                                                        criterion='friedman_mse', min_samples_split=2,
#                                                        min_samples_leaf=1,
#                                                        min_weight_fraction_leaf=0.0, max_depth=3,
#                                                        min_impurity_decrease=0.0, init=None, random_state=None,
#                                                        max_features=None,
#                                                        alpha=0.9, verbose=0, max_leaf_nodes=None, warm_start=False,
#                                                        validation_fraction=0.1, n_iter_no_change=None, tol=0.0001,
#                                                        ccp_alpha=0.0))
#
# hyperparameters = dict(estimator__learning_rate=[0.01, 0.05, 0.1, 0.15, 0.2], estimator__loss=['squared_error', 'absolute_error', 'huber', 'quantile'],
#                      estimator__n_estimators=[10, 20, 50, 100],
#                      estimator__criterion=['friedman_mse', 'squared_error'], estimator__min_samples_split=[2, 3, 5],
#                      estimator__max_depth=[15, 25, 30, 35, 45], estimator__min_samples_leaf=[2, 3, 5],
#                      estimator__min_impurity_decrease=[0, 0.2, 0.4, 0.6, 0.8, 1],
#                      estimator__max_leaf_nodes=[10, 20, 30, 50, 100])
#
# randomized_search = RandomizedSearchCV(model, hyperparameters, random_state=0, n_iter=5, scoring=None,
#                                        n_jobs=2, refit=True, cv=5, verbose=True,
#                                        pre_dispatch='2*n_jobs', error_score='raise', return_train_score=True)
#
# hyperparameters_tuning = randomized_search.fit(X_train, y_train)
# print('Best Parameters = {}'.format(hyperparameters_tuning.best_params_))
#
# tuned_model = hyperparameters_tuning.best_estimator_
tuned_model= MultiOutputRegressor(
                                GradientBoostingRegressor(loss='squared_error', learning_rate=0.15, n_estimators=20, subsample=1.0,
                                                       criterion='squared_error', min_samples_split=5, max_depth=15,
                                                       min_samples_leaf=3,
                                                       min_weight_fraction_leaf=0.0,
                                                       min_impurity_decrease=0.0, init=None, random_state=None,
                                                       max_features=None,
                                                       alpha=0.9, verbose=0, max_leaf_nodes=100, warm_start=False,
                                                       validation_fraction=0.1, n_iter_no_change=None, tol=0.0001,
                                                       ccp_alpha=0.0)
    )
tuned_model.fit(X_train, y_train)
mae_train = mean_absolute_error(y_train, tuned_model.predict(X_train))
mae_test = mean_absolute_error(y_test, tuned_model.predict(X_test))
print("MAE train: {:.4f}".format(mae_train))
print("MAE test: {:.4f}".format(mae_test))























print("Results of GBR (vision only):")
X = H
y = pose[:, :3]

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5)

# model = MultiOutputRegressor(GradientBoostingRegressor(loss='absolute_error', learning_rate=0.1, n_estimators=100, subsample=1.0,
#                                                        criterion='friedman_mse', min_samples_split=2,
#                                                        min_samples_leaf=1,
#                                                        min_weight_fraction_leaf=0.0, max_depth=3,
#                                                        min_impurity_decrease=0.0, init=None, random_state=None,
#                                                        max_features=None,
#                                                        alpha=0.9, verbose=0, max_leaf_nodes=None, warm_start=False,
#                                                        validation_fraction=0.1, n_iter_no_change=None, tol=0.0001,
#                                                        ccp_alpha=0.0))
#
# hyperparameters = dict(estimator__learning_rate=[0.01, 0.05, 0.1, 0.15, 0.2], estimator__loss=['squared_error', 'absolute_error', 'huber', 'quantile'],
#                      estimator__n_estimators=[10, 20, 50, 100],
#                      estimator__criterion=['friedman_mse', 'squared_error'], estimator__min_samples_split=[2, 3, 5],
#                      estimator__max_depth=[15, 25, 30, 35, 45], estimator__min_samples_leaf=[2, 3, 5],
#                      estimator__min_impurity_decrease=[0, 0.2, 0.4, 0.6, 0.8, 1],
#                      estimator__max_leaf_nodes=[10, 20, 30, 50, 100])
#
# randomized_search = RandomizedSearchCV(model, hyperparameters, random_state=0, n_iter=5, scoring=None,
#                                        n_jobs=2, refit=True, cv=5, verbose=True,
#                                        pre_dispatch='2*n_jobs', error_score='raise', return_train_score=True)
#
# hyperparameters_tuning = randomized_search.fit(X_train, y_train)
# print('Best Parameters = {}'.format(hyperparameters_tuning.best_params_))
#
# tuned_model = hyperparameters_tuning.best_estimator_
tuned_model= MultiOutputRegressor(
                                GradientBoostingRegressor(loss='squared_error', learning_rate=0.15, n_estimators=20, subsample=1.0,
                                                       criterion='squared_error', min_samples_split=5, max_depth=15,
                                                       min_samples_leaf=3,
                                                       min_weight_fraction_leaf=0.0,
                                                       min_impurity_decrease=0.0, init=None, random_state=None,
                                                       max_features=None,
                                                       alpha=0.9, verbose=0, max_leaf_nodes=100, warm_start=False,
                                                       validation_fraction=0.1, n_iter_no_change=None, tol=0.0001,
                                                       ccp_alpha=0.0)
    )
tuned_model.fit(X_train, y_train)
mae_train = mean_absolute_error(y_train, tuned_model.predict(X_train))
mae_test = mean_absolute_error(y_test, tuned_model.predict(X_test))
print("MAE train: {:.4f}".format(mae_train))
print("MAE test: {:.4f}".format(mae_test))
