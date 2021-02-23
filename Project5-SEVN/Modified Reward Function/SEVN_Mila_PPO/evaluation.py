import numpy as np
import torch
import utils
from envs import make_vec_envs
import os

def evaluate(actor_critic, ob_rms, env_name, seed, num_processes, eval_log_dir,
             device, custom_gym, save_path):
    eval_envs = make_vec_envs(env_name, seed + num_processes, num_processes,
                              None, eval_log_dir, device, True, custom_gym)

    vec_norm = utils.get_vec_normalize(eval_envs)
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.ob_rms = ob_rms

    eval_episode_rewards = []
    eval_episode_length = []
    eval_episode_success_rate = []

    obs = eval_envs.reset()
    eval_recurrent_hidden_states = torch.zeros(
        num_processes, actor_critic.recurrent_hidden_state_size, device=device)
    eval_masks = torch.zeros(num_processes, 1, device=device)

    while len(eval_episode_rewards) < num_processes*10:
        with torch.no_grad():
            _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                obs,
                eval_recurrent_hidden_states,
                eval_masks,
                deterministic=True)

        # Obser reward and next obs
        obs, _, done, infos = eval_envs.step(action)

        eval_masks = torch.tensor(
            [[0.0] if done_ else [1.0] for done_ in done],
            dtype=torch.float32,
            device=device)

        for info in infos:
            if 'episode' in info.keys():
                eval_episode_rewards.append(info['episode']['r'])
                eval_episode_length.append(info['episode']['l'])
                eval_episode_success_rate.append(info['was_successful_trajectory'])

    eval_envs.close()

    print(" Evaluation using {} episodes: mean reward {:.5f}, mean_length {:.2f}, mean_success {:.2f} \n".format(
        len(eval_episode_rewards), np.mean(eval_episode_rewards), np.mean(eval_episode_length), np.mean(eval_episode_success_rate)))
    if actor_critic.max_eval_success_rate <= np.mean(eval_episode_success_rate):
        actor_critic.max_eval_success_rate = np.mean(eval_episode_success_rate)
        torch.save([
                actor_critic,
                getattr(utils.get_vec_normalize(eval_envs), 'ob_rms', None)
            ], os.path.join(save_path, str(seed)+ "_best_test.pt"))