# Visual-Inertial Odometry Using Homography Matrix

## **Abstract**

The fusion of visual and inertial cues has become popular in computer vision and robotics due to the complementary nature of the two sensing modalities. In this study, we utilized a homography matrix to extract the visual representation from the images gathered from a monocular camera and the information from an inertial measurement unit (IMU). We proposed four machine learning methods including multi-layer perception (MLP), support vector regressor (SVR), random forest regressor (RFR), and gradient boosting regressor (GBR), to estimate the trajectory. By comparing the proposed models, we found that intermediate fusion performs better than early fusion, and in terms of speed, our proposed MLP network is faster than the state-of-the-art methods. Moreover, we did a thorough component analysis and the results demonstrated that the contribution of the homography matrix in pose estimation is promising and more than IMU data. Lastly, we concluded that the proposed method can be a fast yet robust method to implement in real-time unlike more complicated methods using CNN.

## **Dataset**

* IMU data format[^1]:
  
  •	ax:    acceleration in x (m/s<sup>2</sup>)
  
  •	ay:    acceleration in y (m/s<sup>2</sup>)
  
  •	ay:    acceleration in z (m/s<sup>2</sup>)
  
  •	wx:    angular rate around x (rad/s)
  
  •	wy:    angular rate around y (rad/s)
  
  •	wz:    angular rate around z (rad/s)
  
  [^1]: http://www.cvlibs.net/datasets/kitti/raw_data.php


* Pose (ground truth)[^2]:

	Pose for each frame: [R<sub>(3×3)</sub> |T<sub>(3×1)</sub>]
  
  [^2]: http://www.cvlibs.net/datasets/kitti/eval_odometry.php

## Data Pre-processing protocol

1.	Downloaded IMU data from unsynced folder at 100 Hz

2.	Downloaded images from processed (synced + rectified) sequences at 10 Hz

3.	Downloaded pose from ground truth folder at 10 Hz

4.	Calculated relative translational and rotational pose using Pose ground truth

5.	Stacked each two consecutive frames together

6.	Synced the IMU data with stacked frames using corresponding timestamps (every stacked frames corresponds to 10 IMU samples between the consecutive frames)

7.	Calculated the homography matrix for each two consecutive frames using ORB and magsac++

The whole data set after preprocessing:

•	imu.csv containing (22390 * (10*6))

•	pose.csv containing (22390 * 6)

•	homography_matrixes.csv (22390 * 9)

## Method
To begin with, we proposed two different frameworks which are based on early fusion and intermediate fusion respectively. Early fusion or data-level fusion is a traditional way of fusing data to feed it to the analysis system. In this regard, one of the simplest ways of early-stage data fusion is concatenation. This approach could be challenging for the model to extract useful features and estimate the pose and it is expected that it does not lead to an accurate model. The proposed framework using early fusion is shown in Fig. 1. In contrast, the second approach, intermediate fusion, execute the fusion at a decision-making stage after some processing of the inputs. Generally, this approach often results in better performance since the errors from multiple models are more uncorrelated. Nevertheless, recent research reveals that there is no evidence whether the early, late, or intermediate fusion performs better [^3]. The proposed framework using intermediate fusion is shown in Fig. 2.

![image](https://user-images.githubusercontent.com/51737180/209044553-07cea214-f378-473d-a9b5-07d6ea30575d.png)

Fig. 1 Proposed framework (Early fusion)

![image](https://user-images.githubusercontent.com/51737180/209044609-0d7b8f43-4165-433d-aae5-cffae5b30628.png)

Fig. 2 Proposed framework (Intermediate fusion)

[^3]:E. Rublee, V. Rabaud, K. Konolige, and G. Bradski, “ORB: An efficient alternative to SIFT or SURF,” in 2011 International Conference on Computer Vision, Nov. 2011, pp. 2564–2571. doi: 10.1109/ICCV.2011.6126544.

#### MLP

In order to extract the features from the homography matrix and IMU data, MLP is one of the approaches used in this study. MLP is a fully connected class of feedforward neural networks that widely is used in different tasks including classification, regression, etc. Each layer of MLP consists of nodes that are inspired by the neurons in the brain. The network structure is inspired by the work done by Chen et. al [^4], nevertheless, we programmed the whole structure from scratch and also, and we implemented the early fusion as well. The detail of the network is discussed fully in the next section.

[^4]: C. Chen et al., “Selective Sensor Fusion for Neural Visual-Inertial Odometry,” in 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Long Beach, CA, USA, Jun. 2019, pp. 10534–10543. doi: 10.1109/CVPR.2019.01079.

#### RFR, SVR, and GBR

SVR is a supervised learning algorithm that is used to predict discrete values, which uses the same principle as the Support Vector Machine (SVM). The basic idea behind SVR is to find the best fit hyperplane that has the maximum number of points. SVR works relatively well when there is a clear margin of separation between classes, and it is more efficient in high-dimensional spaces. A random forest is a meta estimator that fits several classifying decision trees on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting. RFR is a supervised learning algorithm that uses the ensemble learning method for regression. The ensemble learning method is a technique that combines predictions from multiple machine learning algorithms to make a more accurate prediction than a single model. Since its non-linear feature, RFR outperforms other linear algorithms. Gradient boosting builds an additive model in a forward stage-wise fashion, and it allows for the optimization of arbitrary differentiable loss functions. In each stage, a regression tree is fit on the negative gradient of the given loss function.

![image](https://user-images.githubusercontent.com/51737180/209045473-55bfdc15-ffb0-4034-a895-6913e114e96c.png)

Fig. 3 The architecture of the MLP models. The left one is early fusion and the right one is intermediate fusion.

#### Hyperparameter Tuning and Feature Analysis

Hyper-parameters are parameters that are not directly learned
within estimators. We used the GridSearchCV() function
which exhaustively considers all parameter combinations, to
search the hyper-parameter space for the best crossvalidation
score. We used a 5-fold cross-validation with a
ratio of 0.7:0.3 for the training set & validation set to find the
best model of SVR, RFR, and GBR. We also evaluate the
importance of features on our pose estimation task for the
SVR, RFR, and GBR models. The importance of a feature is
computed as the normalized total reduction of the criterion
brought by that feature. It is also known as the Gini
importance.

## Experiment and Results

In this section, we demonstrated and compared MLP, SVR,
RFR, and GBR. We use the mean absolute error (MAE) for
evaluation. Moreover, to evaluate the prediction qualitatively,
the trajectory was generated using the relative pose. The
trajectory is calculated by cumulating the relative position in
the x and y-direction. During testing, the methods are able to
process the features of the available sensor data and produce
a trajectory at scale. The early fusion and intermediate fusion
of the MLP model are implemented using Keras API and
GPU of Titan Xp.

#### MLP

First, IMU data has been flattened and for the early fusion
approach, the homography matrix and IMU data have been
concatenated. Therefore, the input shape for the early fusion
is a 1 × 69 vector, and the entire architecture for early fusion
and intermediate fusion are shown in Fig. 3. The activation
function in all layers is “Relu” since it makes the model nonlinear
and related works have shown its capabilities in this
field [^4]. Moreover, the loss function for both models is MSE,
and the initial learning rate is 1e-3. Furthermore, by means of
a ReduceLROnPlateau function, the learning rate is reduced
if there is no improvement in the loss after 4 epochs, by the
factor of 0.7. This can help the model further be trained cause
in higher epochs models often benefit from reducing the
learning rate.
In addition, to control the overfitting issue, we utilized l2
regularization as well as a dropout. Indeed, the best solution
to overcome the overfitting problem is to use more data,
nevertheless, in our case, the data is limited to 22390 samples
with the label. Therefore, we tried to expand the model to the
extent that it will be capable of generalization as well as
fitting well. Moreover, the dropout ratio is 0.15, the l2 value
is 1e-5, data is shuffled, and the last layer has no activation
function since it is a regression task. For the batch size, we
tried several numbers from 8 to 1024, and finally, 128 worked
best and was relatively fast as well.
The mean absolute error for the relative translational pose is
shown in table I. Furthermore, the trajectories of the
prediction and ground truth are shown in Fig 4 and 5.

![image](https://user-images.githubusercontent.com/51737180/209046018-2b4d543f-84b8-4954-8b9e-8216dcf2758b.png)

Fig. 4 (a) The trajectory of MLP (early fusion) – (b) The trajectory of
MLP (intermediate fusion). It can be seen that intermediate fusion one
is closer to the ground truth.

![image](https://user-images.githubusercontent.com/51737180/209046039-2033796f-b50c-407e-afc0-6ddb11829c23.png)

Fig. 5 (a) The trajectory of MLP (early fusion) – (b) The trajectory of
MLP (intermediate fusion). The accuracy of the intermediate fusion is
better to some extent than the early fusion one.

#### SVR, RFR, and GBR
We did the grid search to fine-tune the parameters of the SVR model, RFR model, and GBR model. The MAE of the best SVR model, RFR model, and GBR model are shown in Table. 4. We can see that the most superior models in order are RFR, SVR, and GBR, and the vision data make more contributions than inertial data on the three models. Since the RFR model is the best performing model of the three models, its trajectory is shown in Fig. 6.

![image](https://user-images.githubusercontent.com/51737180/209046194-434b832d-ac21-45f4-8391-1af11e5331fb.png)

Fig. 6 The trajectory images for relative pose estimation with different data combinations for the RFR method. (a) With both type of data. (b) With vision data only. (c) With inertial data only.

For the traditional machine learning methods SVR, RFR, and GBR, we can see that the most superior models in order are RFR, SVR, and GBR. Moreover, the vision data make more contributions than inertial data in the three models. Furthermore, we can reduce the number of features based on importance in the fusion data to reduce the need for computation and process time.

In terms of feature importance analysis, since the best SVR model is the ‘RBF’ kernel, we didn’t get the feature importance quantitative results for the SVR model. For the RFR model, we found that the first vision data did the most contribution in the vision part with a 0.0678 score which is rational since the ℎ<sub>11</sub> and ℎ<sub>22</sub> components of the homography matrix are responsible for the change in scale and rotation. Moreover, the second IMU data did the most contribution in the inertial part with a 0.0336 score which corresponds to acceleration in the y-direction (left or right direction in a moving vehicle). For the GBR model, we found the same effects of components of IMU and homography matrix.
As can be seen in Table I, based on the translational error, MLP outperforms others. Indeed, this is expected since the complexity of the MLP neural network is much more than other models and it can better handle the number of features and samples we used. Moreover, in terms of early or intermediate fusion, intermediate fusion can perform better to some extent than early fusion. Furthermore, this superiority of intermediate fusion can be seen in the trajectory as well which the prediction is closer to the ground truth. For instance, in Fig 6(b), the prediction line from x=600 to x=1000 (m) overlies the ground truth, however, in Fig 6(a) the prediction line has offset from the ground truth. Furthermore, it is noteworthy that despite the intermediate fusion model having a lower number of network parameters, it has a better ability to learn. Additionally, in Fig. 5, the offset in the y-direction is more than in the x-direction. Therefore, this issue may be solved by using different weights for the outputs of the network. For example, a higher coefficient for the y component of the loss function can give more priority to this component and force it to be closer to the ground truth. For future work, this can be a possible direction to further improve the accuracy.

Another way to make the model more accurate is using outlier detection algorithms to improve the quality of imu data. This is because the IMU sensor often contains a lot of outliers, and this can lead to worse performance of the neural network. Despite using MAE as a loss function to further reduce the effect of outliers, removing the outliers before feeding them to the network can be another approach to further enhance the performance of the network. This is because the ablation study reveals that the significance of imu data in the learning process is much less than visual data. Table 1 it is shown that using the vision data alone can result in a much better result than using the inertial data alone.

![image](https://user-images.githubusercontent.com/51737180/209047420-249251d9-f12c-4eeb-9c7a-a50cf902e0f3.png)

Lastly, in terms of computation cost, proposed networks have a smaller number of parameters compared to the state-of-the-art model given that the homography computation is fast enough to compute in real-time using the CPU. Therefore, we can conclude that although the error is far from the state-of-the-art model, in terms of speed, our proposed network is faster intuitively. Further quantitative analysis of the speed can be another direction of future work.

