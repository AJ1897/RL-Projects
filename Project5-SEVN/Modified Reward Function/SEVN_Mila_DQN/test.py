"""

### NOTICE ###
You DO NOT need to upload this file

"""
from pyglet.gl import *

import argparse
import numpy as np
import time
import torch
import random
seed = 11037

def parse():
    parser = argparse.ArgumentParser(description="DS595/CS525 RL Project3")
    parser.add_argument('--test_dqn', action='store_true', help='whether test DQN')
    try:
        from argument import add_arguments
        parser = add_arguments(parser)
    except:
        pass
    args = parser.parse_args()
    return args


def test(agent, env, total_episodes=30,current_time=""):
    Path_test_weights = './max_test_weights_dqn_'+ current_time + '.tar'
    rewards = []
    env.seed(seed)
    start_time = time.time()
    success_list=[]
    agent.Test_success_list = []
    agent.Test_reward_list = []
    e_seed=0
    seed_counter = 0

    for i in range(1,total_episodes+1):
        if(i%(total_episodes/10)==0):
            seed_counter+=1
        env.seed(e_seed+seed_counter)
        torch.manual_seed(e_seed+seed_counter)
        np.random.seed(e_seed+seed_counter)
        random.seed(e_seed+seed_counter)
        state = env.reset()
        # if seed_counter == 10:
        #     seed_counter=0
        done = False
        episode_reward = 0.0


        while not done:
            # env.render()
            action = agent.make_action(state, test=True)
            # print(action)
            state, reward, done, info = env.step(action)
            if done:
                if info['was_successful_trajectory']:
                    agent.Test_success_list.append(1)
                else:
                    agent.Test_success_list.append(0)

            episode_reward += reward


        agent.Test_reward_list.append(episode_reward)


    print('Run %d episodes'%(total_episodes))
    print('Mean:', np.mean(agent.Test_reward_list))
    print("Success_rate", np.mean(agent.Test_success_list)*100)
    # print('rewards',agent.Test_reward_list)
    # print('Success_List',agent.Test_success_list)


    if np.mean(agent.Test_success_list)>=agent.max_test_success:
      agent.max_test_success = np.mean(agent.Test_success_list)
      print("max_success = %0.2f"%np.mean(agent.Test_success_list))
      # print("Max_Reward = %0.2f"%np.mean(agent.Test_reward_list))
      print("Saving_Test_Weights_Model")
      torch.save({
        'target_state_dict':agent.Target_DQN.state_dict(),
        'train_state_dict':agent.DQN.state_dict(),
        'optimiser_state_dict':agent.optimiser.state_dict()
        },Path_test_weights)


# def run(args):
#     env = Environment('BreakoutNoFrameskip-v4', args, atari_wrapper=True, test=True)
#     from agent_dqn import Agent_DQN
#     agent = Agent_DQN(env, args)
#     test(agent, env, total_episodes=5)


if __name__ == '__main__':
    args = parse()
    run(args)
