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

## Data Pre-processing protocol:

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


