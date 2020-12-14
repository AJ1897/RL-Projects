# RoboNav: Robot navigation in dynamic environments using Deep Reinforcement Learning
<img src="https://github.com/AJ1897/RL-Projects/blob/master/Project4-RoboNav/Additional_Materials/robotnav.png" width="300">

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

- Launch the DDPG training\
`roslaunch turtlebot3_ddpg turtlebot3_ddpg_torch.launch stage:=3 mode:='train' move_3:='true'`


# TurtleBot3
<img src="https://github.com/ROBOTIS-GIT/emanual/blob/master/assets/images/platform/turtlebot3/logo_turtlebot3.png" width="200">

## Wiki for turtlebot3_machine_learning Packages
- http://wiki.ros.org/turtlebot3_machine_learning (metapackage)
- http://wiki.ros.org/turtlebot3_dqn

## References
- [ROBOTIS e-Manual for TurtleBot3](http://turtlebot3.robotis.com/)
