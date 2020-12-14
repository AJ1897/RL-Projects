# RoboNav: Robot navigation in dynamic environments using Deep Reinforcement Learning
<img src="https://github.com/AJ1897/RL-Projects/blob/master/Project4-RoboNav/Additional_Materials/robotnav.png" width="300">
In this project, we explored an end-to-end learning approach to train a navigation agent from raw perception information (i.e. laser scans) to velocity commands. Specifically, we consider two off-policy learning algorithms, Deep Q Network and Deep Deterministic Policy Gradient and train agents in different simulated training environments to perform point-to-point (P2P) navigation without colliding with obstacles. We evaluate our models against a baseline, Move-Base, which is a well-known classical navigation implementation in ROS. This report discusses our implementation, simulation results, findings and lessons learned

## Setup:
- Ubuntu 18.04
- ROS Melodic
- Gazebo 9
- Pytorch
- Cuda 10.2

## Techniques Used:
- Deep-Q-Network (DQN)
- Deep-Deterministic-Policy-Gradient (DDPG)
- Move-Base 

## Steps to run the RoboNav

- Set the TURTLEBOT3_MODEL in system environment\
`export TURTLEBOT3_MODEL=burger`

- Launch Gazebo with turtlebot3 in Gazebo\
  `roslaunch turtlebot3_gazebo turtlebot3_stage_3.launch`
  
- Launch the DQN training\
`roslaunch turtlebot3_dqn turtlebot3_dqn_torch.launch stage:=3 method:='dueling' mode:='train' move_3:='true'`

- Launch Gazebo with turtlebot3 in stage 1 environment for move-base\
`roslaunch turtlebot3_move_base turtlebot3_stage_1-edit.launch`

- Simulte both move_base & data logger for move-base
`rosrun turtlebot3_move_base `


- Launch the DDPG training\
`roslaunch turtlebot3_ddpg turtlebot3_ddpg_torch.launch stage:=3 mode:='train' move_3:='true'`


# TurtleBot3
<img src="https://github.com/ROBOTIS-GIT/emanual/blob/master/assets/images/platform/turtlebot3/logo_turtlebot3.png" width="200">

## Wiki for turtlebot3_machine_learning Packages
- http://wiki.ros.org/turtlebot3_machine_learning (metapackage)
- http://wiki.ros.org/turtlebot3_dqn

## References
- [ROBOTIS e-Manual for TurtleBot3](http://turtlebot3.robotis.com/)
