from collections import deque
from pathlib import Path

import numpy as np
import torch


def maddpg(env, brain_name, result_dir, agents, memory, n_episodes=7000, max_t=1500):
    """
    Multiple Agent DDPG
    :param env: RL environment
    :param brain_name: Chosen brain
    :param Path result_dir: Path to result dir
    :param list agents: List of Agent objects
    :param memory: Shared replay buffer
    :param int n_episodes: Maximum number of training episodes
    :param int max_t: Maximum number of timesteps per episode
    :return list scores: Scores from each episode
    """

    episode_scores = []
    moving_average_episode_scores = []
    scores_window = deque(maxlen=100)
    num_agents = len(agents)

    for i in range(1, n_episodes):
        env_info = env.reset(train_mode=False)[brain_name]
        [agent.reset() for agent in agents]
        states = env_info.vector_observations
        scores = np.zeros(num_agents)
        for t in range(max_t):
            actions = np.vstack([agent.act(state) for agent, state in zip(agents, states)])
            env_info = env.step(actions)[brain_name]
            next_states, rewards, dones = env_info.vector_observations, env_info.rewards, env_info.local_done
            for num_agent, agent in enumerate(agents):
                agent.step(states[num_agent], actions[num_agent], rewards[num_agent], next_states[num_agent],
                           dones[num_agent], memory)
            scores += rewards
            states = next_states
            if np.any(dones):
                break

        episode_score = np.max(scores)
        episode_scores.append(episode_score)
        scores_window.append(episode_score)
        moving_average_episode_scores.append(np.mean(scores_window))

        print(f'\rEpisode: {i}\tScore (max over agents): {episode_score}', end="")
        if i % 100 == 0:
            print(f'\rEpisode: {i}\tAverage_score: {np.mean(scores_window)}')
        if np.mean(scores_window) >= 0.5:
            print(f'\nEnvironment solved in: {i - 100} episodes!\tAverage_score: {np.mean(scores_window)}')
            for num_agent, agent in enumerate(agents):
                torch.save(agent.actor_local.state_dict(), Path(result_dir) / f'actor_model_player_{num_agent}_solution.pth')
                torch.save(agent.critic_local.state_dict(), Path(result_dir) / f'critic_model_player_{num_agent}_solution.pth')
            break

    return episode_scores, moving_average_episode_scores
