import argparse
from test import test
from environment import Environment
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse():
    parser = argparse.ArgumentParser(description="DS595/CS525 RL Project 3")
    parser.add_argument('--env_name', default=None, help='environment name')
    parser.add_argument('--train_dqn', action='store_true', help='whether train DQN')
    parser.add_argument('--test_dqn', action='store_true', help='whether test DQN')
    parser.add_argument('--cont', action='store_true', help='whether continue DQN')
    parser.add_argument('--n_heads', default=1, help='whether n_heads')
    try:
        from argument import add_arguments
        parser = add_arguments(parser)
    except:
        pass
    args = parser.parse_args()
    return args


def run(args):
    if args.train_dqn:
        env_name = args.env_name or 'BreakoutNoFrameskip-v4'
        env = Environment(env_name, args, atari_wrapper=True)
        from agent_dqn import Agent_DQN
        agent = Agent_DQN(env, args)
        print("Device: ",device)
        print("n_heads: ",args.n_heads)
        agent.train()

    if args.test_dqn:
        env = Environment('BreakoutNoFrameskip-v4', args, atari_wrapper=True, test=True)
        from agent_dqn import Agent_DQN
        agent = Agent_DQN(env, args)
        test(agent, env, total_episodes=10)


if __name__ == '__main__':
    args = parse()
    run(args)
