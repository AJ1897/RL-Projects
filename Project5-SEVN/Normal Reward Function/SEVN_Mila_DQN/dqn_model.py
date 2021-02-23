#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

############################ Convolution Model ############################

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet,self).__init__()
        init_cnn = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init
                                  .constant_(x, 0),
                                  nn.init.calculate_gain('relu'))
        self.conv1 = init_cnn(nn.Conv2d(8,96, kernel_size = 3, stride = 2))
        self.conv2 = init_cnn(nn.Conv2d(96,96,kernel_size = 5, stride = 2))
        self.conv3 = init_cnn(nn.Conv2d(96,32,kernel_size = 5, stride = 2))
        self.norm = nn.BatchNorm2d(8)
        
    def forward(self,x):
        x = self.norm(x.float())
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0),-1)

        return x

############################ Dueling DQN Model  ############################

class DuelNet(nn.Module):
    def __init__(self):
        super(DuelNet,self).__init__()
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))
        init_cnn = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init
                                  .constant_(x, 0),
                                  nn.init.calculate_gain('relu'))
        self.fc = init_cnn(nn.Linear(32*8*8,1024))
        self.val = init_(nn.Linear(512,1))
        self.adv = init_(nn.Linear(512,5))
    def forward(self,x):
        x = F.relu(self.fc(x))
        val,adv = torch.split(x,512,dim=1)
        val = self.val(val)
        adv = self.adv(adv)
        x = val + adv - torch.mean(adv,dim=1,keepdim=True)
        return x

############################ Normal DQN Model  ############################

class LinNet(nn.Module):
    def __init__(self):
        super(LinNet,self).__init__()
        self.fc1 = nn.Linear(32*8*8,512)
        self.fc2 = nn.Linear(512,5)
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

############################ Bootstrap DQN Model  ############################

class BootNet(nn.Module):
    def __init__(self,n_heads, duel = False):
        super(BootNet,self).__init__()
        self.conv_net = ConvNet()
        if duel:
            print("Time for a DuelDQN!")
            self.network_list = nn.ModuleList([DuelNet() for head in range(n_heads)])
        else:
            print("Time for a DQN!")
            self.network_list = nn.ModuleList([LinNet() for head in range(n_heads)])

    def _conv(self,x):
        return self.conv_net(x)
    
    def _lin(self,x):
        return [lin(x) for lin in self.network_list]

    def forward(self,x,head_index=None):
        if head_index is not None:
            return self.network_list[head_index](self._conv(x))
        else:
            conv = self._conv(x)
            return self._lin(conv)

