#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import numpy as np
from collections import deque,Counter
import os
import sys

import gym
import SEVN_gym
from test import test
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
# from agent import Agent
from dqn_model import BootNet
from torch.utils.tensorboard import SummaryWriter
import time
import datetime
"""
you can import any package and define any extra function as you need
"""

torch.manual_seed(595)
np.random.seed(595)
random.seed(595)
current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
# current_time = '2020-12-22-00:03:03'
Path_weights = './last_train_weights_dqn'+current_time+'.tar'
Path_memory = './last_memory_dqn'+current_time+'.tar'
Path_test_weights = './max_test_weights_dqn'+ current_time + '.tar'
tensor_board_dir='./logs/'+current_time
writer = SummaryWriter(tensor_board_dir)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ",device)



class my_dataset(Dataset):
    def __init__(self,data):
        self.samples = data
    def __len__(self):
        return len(self.samples)
    def __getitem__(self,idx):
        return self.samples[idx]


class Agent_DQN(object):
    def __init__(self, env, args):
        """
        Initialize everything you need here.
        For example: 
            paramters for neural network  
            initialize Q net and target Q net
            parameters for repaly buffer
            parameters for q-learning; decaying epsilon-greedy
            ...
        """

        super(Agent_DQN,self).__init__()
        ###########################
        # YOUR IMPLEMENTATION HERE #
        
        ############# Train Parameters #############
        
        self.epochs = 10
        self.args = args
        self.n_episodes = 10000000
        self.env = env
        self.nA = self.env.action_space.n
        self.batch_size = 32
        self.eval_num=0
        self.n_heads = int(self.args.n_heads)
        self.learning_rate = 3e-04
        self.discount_factor = 0.99
        self.Evaluation = 100000
        self.total_evaluation__episodes = 100
        self.full_train = 100000
        
        ############# Model Parameters #############
        self.Duel_DQN = True
        self.Double_DQN = True
        self.DQN = BootNet(self.n_heads,self.Duel_DQN).to(device)
        self.Target_DQN = BootNet(self.n_heads,self.Duel_DQN).to(device)
        self.criteria = nn.SmoothL1Loss()
        self.optimiser = optim.Adam(self.DQN.parameters(),self.learning_rate)

        ############# Buffer Parameters #############
        
        self.buffer_memory = 40000
        self.train_frequency = 4
        self.min_buffer_size = 10000
        self.target_update_buffer = 20000
        self.buffer=[]
        
        ############# Epsilon Parameters #############
        
        self.max_steps = 2000000
        self.annealing_steps = 500000
        self.start_epsilon = 1
        self.end_epsilon_1 = 0.1
        self.end_epsilon_2 = 0.01
        self.slope1 = -(self.start_epsilon - self.end_epsilon_1)/self.annealing_steps
        self.constant1 = self.start_epsilon - self.slope1*self.min_buffer_size
        self.slope2 = -(self.end_epsilon_1 - self.end_epsilon_2)/(self.max_steps - self.annealing_steps - self.min_buffer_size)
        self.constant2 = self.end_epsilon_2 - self.slope2*self.max_steps

        ############# Other Train Parameters #############
        
        self.next_obs = self.env.reset()
        self.done = False
        self.terminal = False
        self.x = 0
        self.ep = 0
        self.current = 0
        self.reward_list =[]
        # self.game_reward=[]
        self.loss_list= []
        self.current_train = 0
        self.current_target = 0
        self.max_test_reward= 0
        self.Test_reward_list = []
        self.Test_success_list = []
        # self.maximum_train_game_reward=0
        self.head_list = list(range(self.n_heads))

        # writer.add_hparams({"Learning_Rate":self.learning_rate,"Batch_Size":self.batch_size,"Discount Factor":self.discount_factor,"Total Episodes":self.n_episodes,"Buffer Size":self.buffer_memory},{"Max__Test_Reward":self.max_test_reward})
        
        ############# Continue Training #############
        
        if args.cont:
          print("#"*50+"Resuming Training"+"#"*50)
          dic_weights = torch.load(Path_weights,map_location=device)
          dic_memory = torch.load(Path_memory)
          self.epsilon = dic_memory['epsilon']
          self.x = dic_memory['x']
          self.max_test_reward = dic_memory['max_test_reward']
          self.ep = dic_memory['ep']
          self.current = dic_memory['current_info'][0]
          self.current_target = dic_memory['current_info'][1]
          self.current_train = dic_memory['current_info'][2]
          self.next_obs = dic_memory['next_info'][0]
          self.done = dic_memory['next_info'][1]
          self.terminal = dic_memory['next_info'][2]
          self.reward_list = []
          self.DQN.load_state_dict(dic_weights['train_state_dict'])
          self.Target_DQN.load_state_dict(dic_weights['target_state_dict'])
          self.DQN.train()
          self.Target_DQN.train()
          self.optimiser.load_state_dict(dic_weights['optimiser_state_dict'])
        
        ############# Testing #############
        
        if args.test_dqn:
            #you can load your model here
            print('loading trained model')
            ###########################
            # YOUR IMPLEMENTATION HERE #
            dic_weights = torch.load(Path_test_weights,map_location=device)
            torch.save(dic_weights['train_state_dict'],'./trained_model_game'+current_time+'.pth')
            dic_weights = torch.load('./trained_model_game'+current_time+'.pth',map_location=device)
            self.DQN.load_state_dict(dic_weights)
            self.DQN.eval()
            

    def init_game_setting(self):
        """
        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary.
        If no parameters need to be initialized, you can leave it as blank.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        # obs = self.env.reset()
        ###########################
        pass
    
    
    def make_action(self, observation, active_head = None, test=True):
        """
        Return predicted action of your agent
        Input:
            observation: np.array shape: (8,84, 84)
        Return:
            action: int
                the predicted action from trained model
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        
        if test:
            # observation = np.transpose(observation,(2,0,1))
            self.epsilon = 0
        
        elif self.current < self.min_buffer_size:
            self.epsilon = 1
        
        elif self.current >= self.min_buffer_size and self.current < self.min_buffer_size + self.annealing_steps:
            self.epsilon = self.current*self.slope1 + self.constant1
        
        elif self.current >= self.min_buffer_size + self.annealing_steps:
            self.epsilon = self.current*self.slope2 + self.constant2
        
        else:
            self.epsilon = 0

        p = np.random.rand()
        if p < self.epsilon:
            action = np.random.randint(0,self.nA)
        else:
            q_values = self.DQN(torch.from_numpy(observation).unsqueeze(0).to(device),active_head)
            if active_head is not None:
                action = torch.argmax(q_values.data,dim = 1).item()
            else:
                acts = [torch.argmax(q_values[k].data,dim = 1).item() for k in range(self.n_heads)]
                data = Counter(acts)
                action = data.most_common(1)[0][0]

        ###########################
        return action
    
    def push(self,episode):
        """ You can add additional arguments as you need. 
        Push new data to buffer and remove the old one if the buffer is full.
        
        Hints:
        -----
            you can consider deque(maxlen = 10000) list
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        if len(self.buffer) < self.buffer_memory:
            self.buffer.append(episode)
        else:
            self.buffer.pop(0)
            self.buffer.append(episode)
        ###########################
        
        
    def replay_buffer(self):
        """ You can add additional arguments as you need.
        Select batch from buffer.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE # 
        batch = random.sample(self.buffer,self.batch_size)
        batch = list(zip(*batch))
        batch_x = torch.from_numpy(np.asarray(batch[0]))
        act = torch.from_numpy(np.asarray(batch[1]))
        rew = torch.from_numpy(np.asarray(batch[2]))
        dones = torch.from_numpy(np.asarray(batch[3])).to(device)
        batch_y = torch.from_numpy(np.asarray(batch[4]))
        mask = torch.from_numpy(np.asarray(batch[5])).to(device)
        ###########################
        return batch_x,act,rew,dones,batch_y,mask
        

    def learn(self):
        
        self.optimiser.zero_grad()
        
        batch_x,actions,rew,dones,batch_y,masks = self.replay_buffer()
        Predicted_q_vals_list = self.DQN(batch_x.to(device))
        Target_q_vals_list = self.Target_DQN(batch_y.to(device))
        Target_policy_vals_list = self.DQN(batch_y.to(device))
        count_losses = []

        for k in range(self.n_heads):
            
            total_used = torch.sum(masks[:,k])
            
            if total_used > 0:
                Target_q_values = Target_q_vals_list[k].data
                
                if (self.Double_DQN):
                    next_actions = Target_policy_vals_list[k].max(1,True)[1]
                    Y = Target_q_values.gather(1,next_actions).squeeze(1)
                else:
                    Y = Target_q_values.max(1,True)[0].squeeze(1)


                Predicted_q_values = Predicted_q_vals_list[k].gather(1,actions[:,None].to(device)).squeeze(1)
                Y[dones] = 0
                Y = Y*self.discount_factor + rew.to(device)

                actual_loss = self.criteria(Predicted_q_values.double(),Y.double())
                propagated_loss = masks[:,k]*actual_loss
                loss = torch.sum(propagated_loss/total_used)
                count_losses.append(loss)

        loss = sum(count_losses)/self.n_heads
        loss.backward()

        for param in self.DQN.conv_net.parameters():
            if param.grad is not None:
                param.grad.data *= 1.0/self.n_heads

        nn.utils.clip_grad_norm_(self.DQN.parameters(),1)
        self.optimiser.step()


    def train(self):
        """
        Implement your training algorithm here
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        print("#"*40 + " Starting Training " + "#"*40)
        ep = self.ep
        success_list = [] 
        episode_length=[]
        seed_counter = 0
        e_seed = 0
        for x in range(self.x,self.n_episodes):
            if seed_counter==10:
                seed_counter = 0
            obs = self.next_obs
            done = self.done
            np.random.shuffle(self.head_list)
            active_head = self.head_list[0]
            accumulated_rewards = 0 # Episodic Reward
            success_episodes = 0
            epl=0
            while not done: 
                
                action = self.make_action(obs,active_head,False)
                next_obs,reward,done,info = self.env.step(action)
                epl+=1
                if done:
                    if info['was_successful_trajectory']:
                        success_episodes=1
                    else:
                        success_episodes=0
                    success_list.append(success_episodes)
                    if len(success_list) > 100:
                        success_list = success_list[-15:]
                    # print("Success!")


                # next_obs = np.transpose(next_obs,(2,0,1))
                masks = np.random.binomial(1,1,self.n_heads) # Masks are for selecting the networks for training on this particular transition
                accumulated_rewards+=reward
                
                self.push([obs,action,reward,done,next_obs,masks])
                self.current+= 1
                self.current_train += 1
                self.current_target += 1
                obs = next_obs
                
                if self.current_train % self.train_frequency == 0 and len(self.buffer) > self.min_buffer_size: # Training Conditions
                    self.learn()
                    self.current_train = 0

                if self.current_target > self.target_update_buffer and len(self.buffer) > self.min_buffer_size:  # Update Target Network
                    self.Target_DQN.load_state_dict(self.DQN.state_dict())
                    self.current_target = 0

                
                if self.current % self.Evaluation == 1: # Test the current network
                    print("\n","#" * 40, "Evaluation number %d"%(self.current/self.Evaluation),"#" * 40)
                    self.eval_num = self.current/self.Evaluation
                    env_name = 'SEVN-Test-AllObs-Shaped-v1'
                    env1 = gym.make(env_name)
                    test(self, env1, total_episodes=10,current_time=current_time)
                    print("#" * 40, "Evaluation Ended!","#" * 40,"\n")
                    
                
                if done: # End the episode 
                    episode_length.append(epl)
                    if len(episode_length) > 100:
                      episode_length=episode_length[-100:]
                    self.reward_list.append(accumulated_rewards)
                    accumulated_rewards = 0
                    ep+=1
                    writer.add_scalars('Train/Episode Reward', {"Reward Mean":np.mean(self.reward_list[-10:]),"Reward Min":np.min(self.reward_list[-10:]),"Reward Max":np.max(self.reward_list[-10:])}, global_step=self.current)
                    writer.add_scalars('Train/Episode Length', {"Episode Length Mean":np.mean(episode_length[-10:]),"Episode Length Min":np.min(episode_length[-10:]),"Episode Length Max":np.max(episode_length[-10:])}, global_step=self.current)
                    writer.add_scalar("Train/Episode Reward Mean",np.mean(self.reward_list[-10:]), global_step=self.current)
                    writer.add_scalar("Train/Episode Length Mean",np.mean(episode_length[-10:]), global_step=self.current)
                    writer.add_scalar("Train/Episode Success Rate",np.mean(success_list[-100:]), global_step=self.current)
            
            self.env.seed(e_seed+seed_counter)
            torch.manual_seed(e_seed+seed_counter)
            np.random.seed(e_seed+seed_counter)
            random.seed(e_seed+seed_counter)
            self.next_obs = self.env.reset()
            seed_counter+=1
            self.done = False
            self.terminal = False
            

            if len(self.reward_list) % 50 == 0:
                self.reward_list = self.reward_list[-10:]
            
            if (x+1)%20 == 0: # Print Data 
                print("Current = %d, episode = %d, Average_reward = %0.2f, epsilon = %0.2f, Success_Rate = %0.2f"%(self.current, ep, np.mean(self.reward_list[-10:]), self.epsilon, np.mean(success_list[-100:])*100))
    
            
            if (x+1)%200 == 0: # Save the models
                print("Saving_Weights_Model")
                torch.save({
                  'target_state_dict':self.Target_DQN.state_dict(),
                  'train_state_dict':self.DQN.state_dict(),
                  'optimiser_state_dict':self.optimiser.state_dict()
                  },Path_weights)
                
                print("Saving_Memory_Info")
                torch.save({
                  'current_info':[self.current,self.current_target,self.current_train],
                  'x':x+1,
                  'ep':ep+1,
                  'max_test_reward': self.max_test_reward,
                  'next_info':[self.next_obs,self.done,self.terminal],
                  'epsilon':self.epsilon,
                  }
                  ,Path_memory)






        
        ###########################
