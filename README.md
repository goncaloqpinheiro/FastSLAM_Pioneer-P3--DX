# FastSLAM Implementation for Pioneer P3-DX

This repository contains a FastSLAM implementation for the Pioneer P3-DX mobile robot, developed as part of the Autonomous Systems (SAut) course at Instituto Superior Técnico.

## Key Features
- FastSLAM implementation in Python/ROS
- Simulation environment using `micro_simulator.py`
- Designed for the Pioneer P3-DX platform
- Includes a built in ArUco detector along with the camera calibration environment
- **Ready-to-use launch configurations**:
  - `cam.launch`: Camera initialization and data capture
  - `fastslam.launch`: Real-time FastSLAM with camera input
  - `fastslam_rviz.launch`: Real-time processing with RViz visualization
  - `fastslam_rosbag.launch`: Offline processing with recorded data
  - `fastslam_rviz_rosbag.launch`: Rosbag playback with RViz visualization

## Project Context
Developed for educational purposes in robotic navigation and simultaneous localization and mapping (SLAM). The implementation demonstrates:
- Particle filtering for pose estimation
- Extended Kalman Filters for landmark tracking
- Real-time performance considerations

![rviz (1)](https://github.com/user-attachments/assets/5ea48d03-5da3-46f5-bbd2-cd245d06afa5)

## Run the Program

To run in the matplotlib environment, use the following command:

```bash
roslaunch fast_slam_saut fastslam_rosbag.launch bag_file:=<path/to/rosbag.bag> rate:=0.5 loop:=false
```
Replace `<path/to/rosbag.bag>` with the name of your rosbag

To run in the rviz environment, use the following command:

```bash
roslaunch fast_slam_saut fastslam_rviz_rosbag.launch bag_file:=<path/to/rosbag.bag> rate:=0.5 loop:=false
```

In order to observe the output of the FastSLAM algorithm in RVIZ, you should also add to RVIZ the following topics:

`/fastslam/particles`

`/fastslam/landmarks`

`/fastslam/trajectory`

`/fastslam/robot`

`/fastslam/uncertainty`

`/fastslam/true_path`  (This is not the true path but in reality the raw robot trajectory from the pose topic)



# Micro-Simulation

**Run the Program**   

To run the microsimulator, use the following command:  

```bash
python3 micro_simulator.py
```
**What It Does**

This Python simulator demonstrates FastSLAM (a SLAM algorithm) with:
- **2D robot simulation** (blue circle)
- **Landmark detection** (red circles = true, green crosses = estimated)
- **Particle filter visualization** (gray dots)
- **Uncertainty ellipses** (green ovals around estimates)
- Two operation modes: automatic circular path or manual control (WASD)

<img src="https://github.com/user-attachments/assets/5d8d7753-b796-4a9d-bdb9-047827ce1390" width="400" alt="Micro-simulation visualization">


