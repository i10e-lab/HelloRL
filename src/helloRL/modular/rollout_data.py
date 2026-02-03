from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch
import torch.utils.data as data

# This isn't currently used by the modular system, but you can use it with the `collect_experiences` function.
# It may be useful for your own custom modules.
@dataclass
class ExperienceData:
    states: torch.Tensor # (n_envs, n_steps, state_space)
    actions: torch.Tensor # (n_envs, n_steps, 1)
    next_states: torch.Tensor # (n_envs, n_steps, state_space)
    rewards: torch.Tensor # (n_envs, n_steps, 1)
    terminateds: torch.Tensor # (n_envs, n_steps, 1)
    truncateds: torch.Tensor # (n_envs, n_steps, 1)
    dones: torch.Tensor # (n_envs, n_steps, 1)

    def __len__(self):
        return len(self.rewards)

    def __getitem__(self, idx):
        # If idx is a slice or integer, apply it to the steps dimension (dim=1) across all envs
        return ExperienceData(
            states=self.states[:, idx],
            actions=self.actions[:, idx],
            next_states=self.next_states[:, idx],
            rewards=self.rewards[:, idx],
            terminateds=self.terminateds[:, idx],
            truncateds=self.truncateds[:, idx],
            dones=self.dones[:, idx]
        )
    
    def flatten(self):
        flat_states = self.states.flatten(0, 1).unsqueeze(0)
        flat_actions = self.actions.flatten(0, 1).unsqueeze(0)
        flat_next_states = self.next_states.flatten(0, 1).unsqueeze(0)
        flat_rewards = self.rewards.flatten(0, 1).unsqueeze(0)
        flat_terminateds = self.terminateds.flatten(0, 1).unsqueeze(0)
        flat_truncateds = self.truncateds.flatten(0, 1).unsqueeze(0)
        flat_dones = self.dones.flatten(0, 1).unsqueeze(0)

        return ExperienceData(
            states=flat_states,
            actions=flat_actions,
            next_states=flat_next_states,
            rewards=flat_rewards,
            terminateds=flat_terminateds,
            truncateds=flat_truncateds,
            dones=flat_dones
        )

@dataclass
class RolloutData(ExperienceData):
    critic_values: torch.Tensor # (n_envs, n_steps, 1)
    log_probs: torch.Tensor # (n_envs, n_steps, 1)

    # Returns and advantages are often completed later. It's acceptable to have them initialized as zero values.
    returns: torch.Tensor # (n_envs, n_steps, 1)
    advantages: torch.Tensor # (n_envs, n_steps, 1)

    def __len__(self):
        return len(self.rewards)

    def __getitem__(self, idx):
        # If idx is a slice or integer, apply it to the steps dimension (dim=1) across all envs
        return RolloutData(
            states=self.states[:, idx],
            actions=self.actions[:, idx],
            next_states=self.next_states[:, idx],
            rewards=self.rewards[:, idx],
            terminateds=self.terminateds[:, idx],
            truncateds=self.truncateds[:, idx],
            dones=self.dones[:, idx],
            critic_values=self.critic_values[:, idx],
            log_probs=self.log_probs[:, idx],
            returns=self.returns[:, idx],
            advantages=self.advantages[:, idx]
        )
    
    def flatten(self):
        flat_states = self.states.flatten(0, 1).unsqueeze(0)
        flat_actions = self.actions.flatten(0, 1).unsqueeze(0)
        flat_next_states = self.next_states.flatten(0, 1).unsqueeze(0)
        flat_rewards = self.rewards.flatten(0, 1).unsqueeze(0)
        flat_terminateds = self.terminateds.flatten(0, 1).unsqueeze(0)
        flat_truncateds = self.truncateds.flatten(0, 1).unsqueeze(0)
        flat_dones = self.dones.flatten(0, 1).unsqueeze(0)
        flat_critic_values = self.critic_values.flatten(0, 1).unsqueeze(0)
        flat_log_probs = self.log_probs.flatten(0, 1).unsqueeze(0)
        flat_returns = self.returns.flatten(0, 1).unsqueeze(0)
        flat_advantages = self.advantages.flatten(0, 1).unsqueeze(0)

        return RolloutData(
            states=flat_states,
            actions=flat_actions,
            next_states=flat_next_states,
            rewards=flat_rewards,
            terminateds=flat_terminateds,
            truncateds=flat_truncateds,
            dones=flat_dones,
            critic_values=flat_critic_values,
            log_probs=flat_log_probs,
            returns=flat_returns,
            advantages=flat_advantages
        )