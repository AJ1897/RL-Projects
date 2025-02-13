# Deep Q-learning Network(DQN)

In this project, an off-policy RL algorithm Deep Q Network(DQN) is implemented to play [Breakout](https://gym.openai.com/envs/Breakout-v0/). This projectis completed in Python 3 using [Pytorch](https://pytorch.org/). The goal of the training was to get averaging reward in 100 episodes over **40 points** in **Breakout**, with OpenAI's Atari wrapper & unclipped reward. For more details, please see the [slides](https://docs.google.com/presentation/d/1CbYqY5DfXQy4crBw489Tno_K94Lgo7QwhDDnEoLYMbI/edit?usp=sharing).

<img src="https://github.com/yingxue-zhang/DS595CS525-RL-Projects/blob/master/Project3/materials/project3.png" width="80%" >

## Leaderboard for class of Fall 2020 (DS-595/CS-525)
  
  **Leaderboard for Breakout-DQN** 
  **Update Date: 11/19/2020 16:00**
  
  | Top | Date | Name | Score |
  | :---: | :---:| :---: | :---: | 
  | 1  | 11/19/2020|Abhishek Jain  | 424.21  |
  | 2  | 11/19/2020|Akshay Sadanandan  | 403  |
  | 3  | 11/19/2020|Dhirajsinh Deshmukh  | 393.37  |
  |4 |11/19/2020 |   Daniel Jeswin Nallathambi      | 335.26  |
  | 5  | 11/18/2020|Sayali Shelke  | 334  |
  |6 | 11/19/2020|Varun Eranki  | 298  |
  | 7  | 11/5/2020|Apiwat Ditthapron  | 194.5  | 
  |8 | 11/18/2020|Panagiotis Argyrakis  | 156.09  |
  |9 | 11/20/2020|Scott Tang  | 153.89  |
  |10 | 11/18/2020|Xinyuan Yang  | 139.11  |
 
 

## Installation
Type the following command to install OpenAI Gym Atari environment in your **virutal environment**.

`$ pip install opencv-python-headless gym==0.10.4 gym[atari]`

Please refer to [OpenAI's page](https://github.com/openai/gym) if you have any problem while installing.

## How to run :
training DQN:
* `$ python main.py --train_dqn`

testing DQN:
* `$ python test.py --test_dqn`

