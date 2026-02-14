import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import uuid
from abc import ABC
from collections import Counter
    
def _generate_title(
        algo_name, env_name, continuous, nb_name,
        num_episodes, mean_score, mean_length, window_length,
        params, agent=None, num_sessions=None):
    """Generate a consistent title for plotting functions."""
    title_parts = [f'{algo_name}\n{env_name}, continuous: {continuous}']
    if num_sessions is not None:
        title_parts.append(f', sessions: {num_sessions}')
    if nb_name:
        title_parts.append(f', {nb_name}')

    # Extract actor, critics, and params from agent if provided
    if agent is not None:
        # Actor class name and params
        actor_class_name = agent.actor.__class__.__name__
        actor_params_str = _params_str(agent.actor.params) if hasattr(agent.actor, 'params') and agent.actor.params is not None else ''
        title_parts.append(f'\n{actor_class_name}{f", {actor_params_str}" if actor_params_str else ""}')

        # Group critics by class name and params
        critic_signatures = []
        for critic in agent.critics:
            critic_class_name = critic.__class__.__name__
            critic_params_str = _params_str(critic.params) if hasattr(critic, 'params') and critic.params is not None else ''
            signature = (critic_class_name, critic_params_str)
            critic_signatures.append(signature)

        # Count occurrences of each unique critic signature
        critic_counts = Counter(critic_signatures)

        # Add critics to title, showing count if > 1
        for (critic_class_name, critic_params_str), count in critic_counts.items():
            multiplier = f' x{count}' if count > 1 else ''
            title_parts.append(f'\n{critic_class_name}{multiplier}{f", {critic_params_str}" if critic_params_str else ""}')

        # Agent class name and params
        agent_class_name = agent.__class__.__name__
        agent_params_str = _params_str(agent.params) if hasattr(agent, 'params') and agent.params is not None else ''
        title_parts.append(f'\n{agent_class_name}{f", {agent_params_str}" if agent_params_str else ""}')

    params_str = _params_str(params)
    title_parts.append(f'\nparams: {params_str}')
    title_parts.append(f'\nepisodes: {num_episodes:.2f}, (mean scores: {mean_score:.2f}, '
                       f'mean length: {mean_length:.2f}) -{window_length} eps')
    return ''.join(title_parts)

def _params_str(params):
    individual_name_strings = []

    for param_name, param_value in params.__dict__.items():
        if isinstance(param_value, ABC):
            parent_class_name = param_value.__class__.__bases__[0].__name__

            name = str(param_value).removeprefix(parent_class_name)

            # if last two characters are '()', remove them
            if name.endswith('()'):
                name = name[:-2]

            individual_name_string = f"{param_name}: {name}"
            individual_name_strings.append(individual_name_string)

    name_string = ''

    current_line_length = 0
    max_line_length = 80

    for i, individual_name_string in enumerate(individual_name_strings):
        if i > 0:
            # check if adding this string would exceed max line length
            if current_line_length + len(individual_name_string) + 2 > max_line_length:
                name_string += ',\n'
                current_line_length = 0
            else:
                name_string += ', '
                current_line_length += 2
        
        name_string += individual_name_string
        current_line_length += len(individual_name_string)
    
    return name_string

def scores_timesteps_to_linspace(scores, timesteps, total_timesteps, increments):
    x_sampled = np.arange(0, total_timesteps, increments)

    # # Create interpolation function
    interp_func = interp1d(
        timesteps, scores,
        kind='linear', bounds_error=False,
        fill_value=(scores[0], scores[-1]))

    # # Get the interpolated values
    y_sampled = interp_func(x_sampled)

    return x_sampled, y_sampled

def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, "valid")

def plot_loss(losses, title=None, ylabel='Loss', ylim=None, average_window=None):
    """
    Plot the loss values.
    :param losses: (list or numpy array) loss values to plot
    :param title: (str) title of the plot
    :param ylabel: (str) label for the y-axis
    :param ylim: (list) [min, max] y-axis limits
    :param average_window: (int) window size for moving average
    """
    plt.figure(figsize=(10, 6))
    
    if average_window is not None:
        losses = moving_average(losses, window=average_window)
    
    plt.plot(losses, label='Loss', color='#1f77b4')
    
    if average_window is not None:
        averages = moving_average(losses, window=average_window)
        plt.plot(averages, label='Average', color='#d62728', linewidth=2)
        plt.legend()
    
    plt.xlabel('Episode')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])
    
    plt.ylabel(ylabel)
    plt.show()

# each session should be a tuple of (scores, timesteps)
def plot_sessions_with_timesteps(
        sessions, title=None, ylabel='Score', ylim=None,
        average_window=None, n_timesteps=None, save_dir=None):
    alpha = (1 / len(sessions)) ** 0.5

    all_scores_and_timesteps = []

    linspaces = []

    plt.figure(figsize=(10, 6))
    
    for i, session in enumerate(sessions):
        scores, timesteps = session

        for j in range(len(scores)):
            all_scores_and_timesteps.append((scores[j], timesteps[j]))

        label = 'Scores' if i == 0 else None
        plt.plot(timesteps, scores, label=label, color='#1f77b4', alpha=alpha)

        # # add plot for average over the surrounding `average_window` episodes
        if average_window is not None:
            averages = moving_average(scores, window=average_window)
            truncated_timesteps = moving_average(timesteps, window=average_window)
            label = 'Average' if i == 0 else None
            plt.plot(truncated_timesteps, averages, label=label, color='#d62728', alpha=alpha, linewidth=2)
            plt.legend()

            ls = scores_timesteps_to_linspace(averages, truncated_timesteps, n_timesteps, increments=64)
            linspaces.append(ls)
            
    all_scores_and_timesteps = sorted(all_scores_and_timesteps, key=lambda x: x[1])

    # now plot the overall average across all runs
    if average_window is not None:
        # each linspace is a tuple of (x_sampled, y_sampled)
        # each x_sampled is the same, so we can just take the first one
        x_sampled = linspaces[0][0]
        y_sampled = np.mean([ls[1] for ls in linspaces], axis=0)

        plt.plot(x_sampled, y_sampled, label='Overall Average', color='#d62728', linewidth=2)
        plt.legend()

    plt.xlabel('Timestep')
    plt.title(title)
    plt.grid(True, alpha=0.3)

    if n_timesteps is not None:
        plt.xlim(0, n_timesteps)

    plt.ylabel(ylabel)
    
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])

    if save_dir is not None:
        # make dir if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

        # saves to a uuid file name in the dir
        filename = f"{uuid.uuid4()}.png"
        plt.savefig(f"{save_dir}/{filename}", dpi=100, bbox_inches='tight')

    plt.show()

def plot_session_with_timesteps(
        scores, timesteps, title=None, ylabel='Score', ylim=None,
        average_window=None, n_timesteps=None, save_dir=None):
    plt.figure(figsize=(10, 6))
    plt.plot(timesteps, scores, label='Scores', color='#1f77b4')

    # # add plot for average over the surrounding `average_window` episodes
    if average_window is not None:
        averages = moving_average(scores, window=average_window)
        truncated_timesteps = moving_average(timesteps, window=average_window)
        plt.plot(truncated_timesteps, averages, label='Average', color='#d62728', linewidth=2)
        plt.legend()
            
    plt.xlabel('Timestep')
    plt.title(title)
    plt.grid(True, alpha=0.3)

    if n_timesteps is not None:
        plt.xlim(0, n_timesteps)

    plt.ylabel(ylabel)
    
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])

    plt.show()

    if save_dir is not None:
        # saves to a uuid file name in the dir
        filename = f"{uuid.uuid4()}.png"
        plt.savefig(f"{save_dir}/{filename}", dpi=100, bbox_inches='tight')

def ylim_for_env(env_name, continuous=False):
    """
    Returns the y-axis limits for the given environment.
    :param env_name: (str) name of the environment
    :param continuous: (bool) whether the environment is continuous
    :return: (list) [min, max] y-axis limits
    """
    if env_name == 'CartPole-v1':
        return [0, 600]
    elif env_name == 'MountainCar-v0':
        return [-200, 0]
    elif env_name == 'Acrobot-v1':
        return [-500, 0]
    elif env_name == 'LunarLander-v3':
        if continuous:
            return [-600, 600]
        else:
            return [-600, 600]
    else:
        return [0, 1000]  # default limits for other environments
    
def plot_session(
        scores, episode_lengths, algo_name, env_name, continuous=False, nb_name=None, n_timesteps=1000,
        params=None, agent=None, window_length=64, save_dir=None):
    mean_score = np.mean(scores[-window_length:]) if len(scores) >= window_length else np.mean(scores)
    mean_length = np.mean(episode_lengths[-window_length:]) if len(episode_lengths) >= window_length else np.mean(episode_lengths)

    timesteps = np.cumsum(episode_lengths)

    title = _generate_title(
        algo_name, env_name, continuous, nb_name,
        num_episodes=len(scores),
        mean_score=mean_score,
        mean_length=mean_length,
        window_length=window_length,
        params=params,
        agent=agent
    )

    plot_session_with_timesteps(
        scores, timesteps,
        title=title,
        ylim=ylim_for_env(env_name, continuous=continuous),
        average_window=window_length,
        n_timesteps=n_timesteps,
        save_dir=save_dir)
    
def plot_sessions(
        all_results, algo_name, env_name, nb_name=None, continuous=False, n_timesteps=1000,
        params=None, agent=None, window_length=64, save_dir=None):
    all_scores = [scores for scores, _ in all_results]
    all_episode_lengths = [episode_lengths for _, episode_lengths in all_results]
    # for example, if there are 16 sessions
    # all_scores is a list of 16 lists of scores
    # all_episode_lengths is a list of 16 lists of episode lengths
    # all_results is a list of 16 tuples (episode_rewards, episode_lengths)

    all_timesteps = [np.cumsum(episode_lengths) for episode_lengths in all_episode_lengths]
    sessions = [(all_scores[i], all_timesteps[i]) for i in range(len(all_scores))]

    flattened_scores = [score for scores in all_scores for score in scores]

    last_scores_mean = np.mean([np.mean(scores[-window_length:]) for scores in all_scores])
    last_episode_lengths_mean = np.mean([np.mean(episode_lengths[-window_length:])
        for episode_lengths in all_episode_lengths])

    title = _generate_title(
        algo_name, env_name, continuous, nb_name,
        num_episodes=len(flattened_scores),
        mean_score=last_scores_mean,
        mean_length=last_episode_lengths_mean,
        window_length=window_length,
        params=params,
        agent=agent,
        num_sessions=len(all_results)
    )

    plot_sessions_with_timesteps(
        sessions,
        title=title,
        ylim=ylim_for_env(env_name, continuous=continuous),
        average_window=window_length,
        n_timesteps=n_timesteps,
        save_dir=save_dir)