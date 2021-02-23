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


def main():
    args = get_args()
    l_dir = "./logs/"+args.env_name+'/'+current_time
    writer = SummaryWriter(l_dir)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, args.log_dir, device, False, args.custom_gym)

    base=SEVN

    actor_critic = Policy(
        envs.observation_space.shape,
        envs.action_space,
        base = base,
        base_kwargs={'recurrent': args.recurrent_policy}
        )
    actor_critic.to(device)
    actor_critic.max_eval_success_rate = 0
    print("Passed!")

    if args.algo == 'ppo':
        agent = PPO(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)

    obs = envs.reset()
    print(obs.size())
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    
    if "Train" in args.env_name:
        test_envs = make_vec_envs(args.env_name.replace("Train", "Test"), args.seed, 1,
                             args.gamma, log_dir, device, False,
                             args.custom_gym)
        test_rollouts = RolloutStorage(args.num_steps, 1,
                                       envs.observation_space.shape, envs.action_space,
                                       actor_critic.recurrent_hidden_state_size)
        test_obs = test_envs.reset()
        test_rollouts.obs[0].copy_(test_obs)
        test_rollouts.to(device)
        test_episode_rewards = deque(maxlen=10)
        test_episode_length = deque(maxlen=10)
        test_episode_success_rate = deque(maxlen=100)
        test_episode_total=0

    episode_rewards = deque(maxlen=10)
    episode_length = deque(maxlen=10)
    episode_success_rate = deque(maxlen=100)
    episode_total=0

    start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes
    for j in range(num_updates):

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,args.lr)

        for step in range(args.num_steps):
            # Sample actions
            # print(rollouts.obs[step].size())
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)

            for idx, info in enumerate(infos):
                if 'episode' in info.keys():
                    if info['was_successful_trajectory']:
                        if args.mod: #Modified Reward Function
                            reward[idx]=10
                            episode_rewards.append(info['episode']['r']+10)
                    else:
                        episode_rewards.append(info['episode']['r'])
                    episode_length.append(info['episode']['l'])
                    episode_success_rate.append(info['was_successful_trajectory'])
                    
                    episode_total+=1

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()


        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        if "Train" in args.env_name:
            for step in range(args.num_steps):
                # Sample actions
                with torch.no_grad():
                    value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                        test_rollouts.obs[step], test_rollouts.recurrent_hidden_states[step],
                        test_rollouts.masks[step])

                    # Observe reward and next obs
                    obs, reward, done, infos = test_envs.step(action)
                    for idx, info in enumerate(infos):
                        if 'episode' in info.keys():
                            test_episode_rewards.append(info['episode']['r'])
                            test_episode_length.append(info['episode']['l'])
                            test_episode_success_rate.append(info['was_successful_trajectory'])
                            test_episode_total += 1
                    masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
                    bad_masks = torch.FloatTensor([[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos])
                    test_rollouts.insert(obs, recurrent_hidden_states, action,
                                action_log_prob, value, reward, masks, bad_masks)

                    with torch.no_grad():
                        next_value = actor_critic.get_value(
                            test_rollouts.obs[-1], test_rollouts.recurrent_hidden_states[-1],
                            test_rollouts.masks[-1]).detach()
                        test_rollouts.after_update()



        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0
                or j == num_updates - 1) and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo, args.env_name, current_time)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            torch.save([
                actor_critic,
                getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
            ], os.path.join(save_path, "Last_train.pt"))

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            writer.add_scalars('Train/Episode Reward', {"Reward Mean":np.mean(episode_rewards),"Reward Min":np.min(episode_rewards),"Reward Max":np.max(episode_rewards)}, global_step=total_num_steps)
            writer.add_scalars('Train/Episode Length', {"Episode Length Mean":np.mean(episode_length),"Episode Length Min":np.min(episode_length),"Episode Length Max":np.max(episode_length)}, global_step=total_num_steps)
            writer.add_scalar("Train/Episode Reward Mean",np.mean(episode_rewards), global_step=total_num_steps)
            writer.add_scalar("Train/Episode Length Mean",np.mean(episode_length), global_step=total_num_steps)
            writer.add_scalar("Train/Episode Success Rate",np.mean(episode_success_rate), global_step=total_num_steps)
            if "Train" in args.env_name:
                writer.add_scalars('Test/Episode Reward', {"Reward Mean":np.mean(test_episode_rewards),"Reward Min":np.min(test_episode_rewards),"Reward Max":np.max(test_episode_rewards)}, global_step=total_num_steps)
                writer.add_scalars('Test/Episode Length', {"Episode Length Mean":np.mean(test_episode_length),"Episode Length Min":np.min(test_episode_length),"Episode Length Max":np.max(test_episode_length)}, global_step=total_num_steps)
                writer.add_scalar("Test/Episode Reward Mean",np.mean(test_episode_rewards), global_step=total_num_steps)
                writer.add_scalar("Test/Episode Length Mean",np.mean(test_episode_length), global_step=total_num_steps)
                writer.add_scalar("Test/Episode Success Rate",np.mean(test_episode_success_rate), global_step=total_num_steps)

            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                .format(j, total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards), dist_entropy, value_loss,
                        action_loss))

        if (args.eval_interval is not None and len(episode_rewards) > 1
                and j % args.eval_interval == 0):
            # ob_rms = utils.get_vec_normalize(envs).ob_rms
            evaluate(actor_critic, 0, args.env_name, args.seed,
                     args.num_processes, eval_log_dir, device, args.custom_gym, save_path)
            evaluate(actor_critic, 0, args.env_name, args.seed + args.num_processes,
                     args.num_processes, eval_log_dir, device, args.custom_gym, save_path)
            evaluate(actor_critic, 0, args.env_name, args.seed + args.num_processes*2,
                     args.num_processes, eval_log_dir, device, args.custom_gym, save_path)


if __name__ == "__main__":
    main()
