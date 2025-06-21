# FastSLAM Implementation for Pioneer P3-DX

This repository contains a FastSLAM implementation for the Pioneer P3-DX mobile robot, developed as part of the Autonomous Systems (SAut) course at Instituto Superior Técnico.

## Key Features
- FastSLAM implementation in Python/ROS
- Simulation environment using `micro_simulator.py`
- Designed for the Pioneer P3-DX platform
- Includes sensor noise modeling and data association
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
- Two operation modes: automatic circular path or manual control
