import copy
import glob
import os
import time
from collections import deque

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import utils
from PPO import PPO
from arguments import get_args
from envs import make_vec_envs
from model import Policy, SEVN
from storage import RolloutStorage
from evaluation import evaluate
from torch.utils.tensorboard import SummaryWriter
import datetime

current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
# save_dir = '/home/srilekhawpi/SEVN_MILA/SEVN_Mila_PPO/trained_models/ppo/SEVN-Test-AllObs-Shaped-v1/2020-12-30-16_21_13/Last_train.pt' # Test, Reward = 0.1
save_dir = '/home/srilekhawpi/SEVN_MILA/SEVN_Mila_PPO/trained_models/ppo/SEVN-Test-AllObs-Shaped-v1/2020-12-31-09_36_42/Last_train.pt' # Test, Reward = 10
# save_dir = '/home/srilekhawpi/SEVN_MILA/SEVN_Mila_PPO/trained_models/ppo/SEVN-Train-AllObs-Shaped-v1/2020-12-28-00_24_37/Last_train.pt' # Train, Reward = 0.1
# save_dir = '/home/srilekhawpi/SEVN_MILA/SEVN_Mila_PPO/trained_models/ppo/SEVN-Train-AllObs-Shaped-v1/2020-12-31-09_34_34/Last_train.pt' # Train, Reward = 10

def main():
    args = get_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")


    base=SEVN

    actor_critic, obs_rms = torch.load(save_dir, map_location=device)
    actor_critic.to(device)
    actor_critic.max_eval_success_rate = 0
    print("Passed!")
    num_processes = args.num_processes
    eval_recurrent_hidden_states = torch.zeros(
        args.num_processes, actor_critic.recurrent_hidden_state_size, device=device)
    eval_masks = torch.zeros(num_processes, 1, device=device)
    x = 0
    while x < 10:
        torch.manual_seed(args.seed + x)
        torch.cuda.manual_seed_all(args.seed + x)
        eval_envs = make_vec_envs(args.env_name, args.seed + x, args.num_processes,
                         args.gamma, args.log_dir, device, False, args.custom_gym)
        eval_episode_rewards = []
        eval_episode_length = []
        eval_episode_success_rate = []
        obs = eval_envs.reset()
        while len(eval_episode_rewards) < num_processes*100:
            with torch.no_grad():
                _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                    obs,
                    eval_recurrent_hidden_states,
                    eval_masks,
                    deterministic=True)
            eval_envs.render()
            obs, _, done, infos = eval_envs.step(action)

            eval_masks = torch.tensor(
                [[0.0] if done_ else [1.0] for done_ in done],
                dtype=torch.float32,
                device=device)

            for info in infos:
                if 'episode' in info.keys():
                    if info['was_successful_trajectory']:
                        if args.mod: #Modified Reward Function
                            reward[idx]=10
                            episode_rewards.append(10)
                    else:
                        eval_episode_rewards.append(info['episode']['r'])
                    eval_episode_length.append(info['episode']['l'])
                    eval_episode_success_rate.append(info['was_successful_trajectory'])
        x+=1
        print(" Evaluation using {} episodes: mean reward {:.5f}, mean_length {:.2f}, mean_success {:.2f} \n".format(
        len(eval_episode_rewards), np.mean(eval_episode_rewards), np.mean(eval_episode_length), np.mean(eval_episode_success_rate)))    

    eval_envs.close()

    print(eval_episode_rewards)
    print(eval_episode_success_rate)

if __name__ == "__main__":
    main()
