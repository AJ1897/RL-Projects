import argparse
from test import test
# from environment import Environment
import torch
import SEVN_gym
import gym
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse():
    parser = argparse.ArgumentParser(description="SEVN_Mila_Assignment")
    parser.add_argument('--env_name', default=None, help='environment name')
    parser.add_argument('--train_dqn', action='store_true', help='whether train DQN')
    parser.add_argument('--test_dqn', action='store_true', help='whether test DQN')
    parser.add_argument('--cont', action='store_true', help='whether continue DQN')
    parser.add_argument('--n_heads', default=1, help='whether n_heads')
    parser.add_argument('--mod', action='store_true', help='whether modify reward function')
    try:
        from argument import add_arguments
        parser = add_arguments(parser)
    except:
        pass
    args = parser.parse_args()
    return args


def run(args):
    if args.train_dqn:
        env_name = args.env_name or 'SEVN-Test-AllObs-Shaped-v1'
        env = gym.make(env_name)
        from agent_dqn import Agent_DQN
        agent = Agent_DQN(env, args)
        print("Device: ",device)
        print("n_heads: ",args.n_heads)
        agent.train()

    if args.test_dqn:
        current_time = '2020-12-25-09:13:18'
        env_name = args.env_name or 'SEVN-Test-AllObs-Shaped-v1'
        env = gym.make(env_name)
        from agent_dqn import Agent_DQN
        agent = Agent_DQN(env, args)
        test(agent, env, 10, current_time)


if __name__ == '__main__':
    args = parse()
    run(args)
