from dataclasses import dataclass
import torch

from helloRL.modular.foundation import *
    
@dataclass
class AdvantageTransformNormalize(AdvantageTransform):
    # mean or sum normalization
    method: str = "mean"

    def transform(self, raw_advantages: torch.Tensor) -> torch.Tensor:
        if self.method == "sum":
            return raw_advantages / (raw_advantages.sum() + 1e-9)
        else:  # default to mean normalization
            if len(raw_advantages) < 2:
                return raw_advantages

            return (raw_advantages - raw_advantages.mean()) / (raw_advantages.std() + 1e-9)

@dataclass
class RolloutMethodA2C(RolloutMethod):
    n_steps: int = 16
    n_envs: int = 4

    def collect_rollout_data(
        self, envs: gym.vector.VectorEnv, initial_states: torch.Tensor, agent: AgentProtocol, tracker: SessionTracker
        ) -> tuple[RolloutData, list[np.ndarray]]:
            n_envs = envs.num_envs
            state_space = envs.single_observation_space.shape[0]
            action_dim = envs.single_action_space.shape[0] if isinstance(envs.single_action_space, gym.spaces.Box) else 1

            rollout_states_t = torch.zeros(n_envs, self.n_steps, state_space) # (n_envs, n_steps, state_space)
            rollout_actions_t = torch.zeros(n_envs, self.n_steps, action_dim) # (n_envs, n_steps, action_dim)
            rollout_next_states_t = torch.zeros(n_envs, self.n_steps, state_space) # (n_envs, n_steps, state_space)
            rollout_rewards_t = torch.zeros(n_envs, self.n_steps, 1) # (n_envs, n_steps, 1 value)
            rollout_terminateds_t = torch.zeros(n_envs, self.n_steps, 1) # (n_envs, n_steps, 1 value)
            rollout_truncateds_t = torch.zeros(n_envs, self.n_steps, 1) # (n_envs, n_steps, 1 value)
            rollout_dones_t = torch.zeros(n_envs, self.n_steps, 1) # (n_envs, n_steps, 1 value)
            rollout_critic_values_t = torch.zeros(n_envs, self.n_steps, 1) # (n_envs, n_steps, 1 value)
            rollout_log_probs_t = torch.zeros(n_envs, self.n_steps, 1) # (n_envs, n_steps, 1 value)

            states = initial_states

            for step in range(self.n_steps):
                tracker.increment_timestep(n=n_envs)

                states_t = torch.tensor(states).float() # (n_envs, state_space)

                with torch.no_grad():
                    actions_t, _ = agent.actor.output(states_t)
                    # apply exploration noise etc to the actions
                    actions_t = agent.actor.exploration(actions_t)
                    log_probs_t, _ = agent.actor.get_log_prob_and_entropy(states_t, actions_t)
                    critic_values_t = agent.get_critic_value(states_t, actions_t)
                actions_np = actions_t.squeeze(-1).numpy() # (n_envs)
                next_states, rewards, terminateds, truncateds, infos = envs.step(actions_np)

                dones = terminateds | truncateds
                dones_t = torch.tensor(dones).float().reshape(n_envs, 1)  # (n_envs, 1 value)
                terminateds_t = torch.tensor(terminateds).float().reshape(n_envs, 1)  # (n_envs, 1 value)
                truncateds_t = torch.tensor(truncateds).float().reshape(n_envs, 1)  # (n_envs, 1 value)

                rewards_t = torch.tensor(rewards).float().reshape(n_envs, 1)  # (n_envs, 1 value)
                
                rollout_states_t[:, step] = states_t  # (n_envs, n_steps, state_space)
                rollout_actions_t[:, step] = actions_t  # (n_envs, n_steps, 1 value)
                rollout_next_states_t[:, step] = torch.tensor(next_states).float()  # (n_envs, n_steps, state_space)
                rollout_rewards_t[:, step] = rewards_t  # (n_envs, n_steps, 1 value)
                rollout_terminateds_t[:, step] = terminateds_t  # (n_envs, n_steps, 1 value)
                rollout_truncateds_t[:, step] = truncateds_t  # (n_envs, n_steps, 1 value)
                rollout_dones_t[:, step] = dones_t  # (n_envs, n_steps, 1 value)
                rollout_critic_values_t[:, step] = critic_values_t  # (n_envs, n_steps, 1 value)
                rollout_log_probs_t[:, step] = log_probs_t  # (n_envs, n_steps, 1 value)

                if dones.any():
                    episode_returns = infos['episode']['r'][dones]
                    episode_lengths = infos['episode']['l'][dones]

                    tracker.finish_episodes(episode_returns, episode_lengths)
                    
                    next_states, _ = envs.reset(options={'reset_mask': dones})

                states = next_states
            
            rollout_returns_t = torch.zeros_like(rollout_rewards_t)  # (n_envs, n_steps, 1)
            rollout_advantages_t = torch.zeros_like(rollout_rewards_t)  # (n_envs, n_steps, 1)

            rollout_data = RolloutData(
                states=rollout_states_t,
                actions=rollout_actions_t,
                next_states=rollout_next_states_t,
                rewards=rollout_rewards_t,
                terminateds=rollout_terminateds_t,
                truncateds=rollout_truncateds_t,
                dones=rollout_dones_t,
                critic_values=rollout_critic_values_t,
                log_probs=rollout_log_probs_t,
                returns=rollout_returns_t,
                advantages=rollout_advantages_t
            )

            return rollout_data, next_states