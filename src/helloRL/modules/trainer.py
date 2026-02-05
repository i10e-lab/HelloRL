import gymnasium as gym
import torch
import torch.optim as optim

from helloRL.modules.replay import ExperienceBuffer
from helloRL.modules.foundation import *
from helloRL.modules.params import Params

def make_env(gym_id, continuous=False):
    def thunk():
        if continuous:
            # some envs don't support the continuous parameter at all, so only call it when needed
            env = gym.make(gym_id, continuous=True)
        else:
            env = gym.make(gym_id)

        env = gym.wrappers.RecordEpisodeStatistics(env)

        return env
    return thunk

# next critic values (n_envs, n_steps, 1)
def calculate_returns(rollout_data: RolloutData, next_critic_values: torch.Tensor, gamma: float):
    # our goal with this function is to fill out the 'returns' tensor
    returns = torch.zeros_like(rollout_data.rewards)  # (n_envs, n_steps, 1)

    bootstrapped_returns = next_critic_values
    next_returns = bootstrapped_returns[:, -1, :]  # (n_envs, 1 value)

    dones = rollout_data.dones  # (n_envs, n_steps, 1)

    _, n_steps, _ = rollout_data.rewards.shape

    for step in reversed(range(n_steps)):
        # only applies future returns if not terminated or truncated
        step_future_returns = next_returns * (1 - dones[:, step, :])  # (n_envs, 1)

        # apply bootstrapped returns if truncated
        step_future_returns += bootstrapped_returns[:, step, :] * rollout_data.truncateds[:, step, :]  # (n_envs, 1)

        # apply discounting
        step_future_returns *= gamma  # (n_envs, 1)

        step_returns = rollout_data.rewards[:, step, :] + step_future_returns  # (n_envs, 1)
        returns[:, step, :] = step_returns  # (n_envs, 1)

        next_returns = step_returns  # (n_envs, 1)

    return returns

from helloRL.modules.monte_carlo import RolloutMethodMonteCarlo

def collect_experiences(actor, env_name, continuous=False, n_timesteps=100000, should_print=True, gamma=0.99):
    envs = [make_env(env_name, continuous=continuous)]
    envs = gym.vector.SyncVectorEnv(envs)

    experience_buffer = ExperienceBuffer(capacity=n_timesteps, state_dim=envs.single_observation_space.shape[0], action_dim=envs.single_action_space.shape[0]) 

    next_states, _ = envs.reset()

    rollout_method = RolloutMethodMonteCarlo()

    with SessionTracker(n_timesteps=n_timesteps, should_print=should_print, print_interval=1000, window_length=64) as tracker:
        while not tracker.is_session_complete():
            rollout_data = rollout_method.collect_experience_data(
                envs=envs, initial_states=next_states, actor=actor, tracker=tracker)
            
            # zeroes
            next_critic_values = torch.zeros_like(rollout_data.rewards)
            
            returns = calculate_returns(
                rollout_data, next_critic_values=next_critic_values, gamma=gamma)  # (n_envs, n_steps, 1)
            rollout_data.returns = returns

            experience_buffer.add(rollout_data)

    return experience_buffer.all()


def train(agent, env_name, continuous=False, params:Params=Params(), n_timesteps=100000, should_print=True, seed=None, progress_callback=None):
    n_envs = params.rollout_method.n_envs

    envs = [make_env(env_name, continuous=continuous) for _ in range(n_envs)]
    envs = gym.vector.SyncVectorEnv(envs)

    next_states, _ = envs.reset(seed=seed)

    initial_lr = params.lr_schedule.get_lr(step=0, total_steps=n_timesteps)
    actor_optimizer = optim.Adam(agent.actor.network.parameters(), lr=initial_lr)
    
    critic_optimizers = []

    for critic in agent.critics:
        critic_optimizer = optim.Adam(critic.network.parameters(), lr=initial_lr)
        critic_optimizers.append(critic_optimizer)

    with SessionTracker(
        n_timesteps=n_timesteps, should_print=should_print,
        print_interval=1000, window_length=64, progress_callback=progress_callback) as tracker:
        while not tracker.is_session_complete():
            current_timestep = tracker.current_timestep
            params.lr_schedule.apply(actor_optimizer, step=current_timestep, total_steps=n_timesteps)
            for critic_optimizer in critic_optimizers:
                params.lr_schedule.apply(critic_optimizer, step=current_timestep, total_steps=n_timesteps)

            rollout_data, next_states = params.rollout_method.collect_rollout_data(
                envs=envs, initial_states=next_states, agent=agent, tracker=tracker)
            scaled_rewards = params.reward_transform.transform(rollout_data.rewards)
            rollout_data.rewards = scaled_rewards

            # bootstrap the future returns from the critic, across all envs and timesteps
            with torch.no_grad():
                # (n_envs, n_steps, 1)
                next_actions = agent.get_target_action(rollout_data.next_states.float())
                next_critic_values = agent.get_target_critic_value(rollout_data.next_states.float(), next_actions.float())

            returns = calculate_returns(
                rollout_data, next_critic_values=next_critic_values, gamma=params.gamma)  # (n_envs, n_steps, 1)
            rollout_data.returns = returns

            advantages = params.advantage_method.compute_advantage(rollout_data, next_critic_values, params.gamma)
            advantages = params.advantage_transform.transform(advantages)
            rollout_data.advantages = advantages

            for batch_data in params.data_load_method(rollout_data):
                for i, critic in enumerate(agent.critics):
                    critic_loss = critic.get_loss(
                        target_action_func=agent.get_target_action,
                        target_critic_value_func=agent.get_target_critic_value,
                        data=batch_data,
                        gamma=params.gamma
                    )

                    critic_optimizer = critic_optimizers[i]
                    critic_optimizer.zero_grad()
                    critic_loss.backward()
                    params.gradient_transform.apply(critic.network.parameters())
                    critic_optimizer.step()

                actor_loss = agent.actor.get_loss(batch_data, agent.get_critic_value)
                
                actor_optimizer.zero_grad()
                actor_loss.backward()
                params.gradient_transform.apply(agent.actor.network.parameters())
                actor_optimizer.step()

                agent.update_targets()

        envs.close()

    return tracker.all_returns, tracker.all_lengths