import torch
from dataclasses import dataclass

from helloRL.modular.foundation import *

@dataclass
class AdvantageMethodSingleStep(AdvantageMethod):
    def compute_advantage(
        self, data: RolloutData, next_critic_values: torch.Tensor, gamma: float) -> torch.Tensor:

        critic_current_returns = data.critic_values  # (n_envs, n_steps, 1)
        critic_future_returns = next_critic_values  # (n_envs, n_steps, 1)
        critic_future_returns *= (1 - data.terminateds)  # zero out future returns where episode terminated

        single_step_returns = data.rewards + (gamma * critic_future_returns)  # (n_envs, n_steps, 1)
        advantages = single_step_returns - critic_current_returns.detach()  # (n_envs, n_steps, 1)

        return advantages

@dataclass
class AdvantageMethodGAE(AdvantageMethod):
    lambda_: float = 0.95

    def compute_advantage(
        self, data: RolloutData, next_critic_values: torch.Tensor, gamma: float) -> torch.Tensor:

        # our goal is to compute advantages using generalized advantage estimation
        # GAE computes advantages by accumulating temporal difference errors backwards through time
        # the temporal difference error at each step is: reward + discounted future value - current value
        # then we accumulate these errors with exponential decay controlled by lambda

        # use the single step method to compute single step advantages
        delta = AdvantageMethodSingleStep().compute_advantage(
            data, next_critic_values, gamma)  # (n_envs, n_steps, 1)

        # now we accumulate advantages backwards through time
        # start with empty advantages
        advantages = torch.zeros_like(delta)  # (n_envs, n_steps, 1)

        next_advantage = torch.zeros_like(delta[:, 0, :])  # (n_envs, 1)

        # compute the discount factor for GAE accumulation
        gae_discount = gamma * self.lambda_
        
        _, n_steps, _ = data.rewards.shape

        # iterate backwards through each step
        for step in reversed(range(n_steps)):
            # get the temporal difference error for this step
            step_delta = delta[:, step, :]  # (n_envs, 1)

            # only use future advantages if the episode hasn't ended at this step
            actualised_next_advantage = next_advantage * (1 - data.dones[:, step, :])  # (n_envs, 1)

            # accumulate advantage: current error plus discounted future advantages
            step_advantage = step_delta + (gae_discount * actualised_next_advantage)  # (n_envs, 1)
            advantages[:, step, :] = step_advantage

            # update for next iteration
            next_advantage = step_advantage

        return advantages