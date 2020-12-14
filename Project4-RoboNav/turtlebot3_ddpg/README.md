## Steps to run the RoboNav

- Set the TURTLEBOT3_MODEL in system environment\
`export TURTLEBOT3_MODEL=burger`

- Launch Gazebo with turtlebot3 in Gazebo\
`roslaunch turtlebot3_gazebo turtlebot3_stage_3.launch`

- Launch the DDPG testing\
`roslaunch turtlebot3_ddpg turtlebot3_ddpg_torch.launch stage:=3 mode:='test' move_3:='true'`
