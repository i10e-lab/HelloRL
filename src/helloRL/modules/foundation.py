# rollout_strategy.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch
import torch.optim as optim
import numpy as np
import gymnasium as gym
import torch.nn as nn
from typing import Iterable, Iterator

from helloRL.utils.session_tracker import SessionTracker
from .agents import AgentProtocol
from .rollout_data import RolloutData

@dataclass
class LRSchedule(ABC):
    @abstractmethod
    def get_lr(self, step: int, total_steps: int) -> float:
        """Return the learning rate to use at this global step."""
        pass

    def apply(self, optimizer: optim.Optimizer, step: int, total_steps: int) -> float:
        lr = float(self.get_lr(step, total_steps))
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        return lr
    
@dataclass
class LRScheduleConstant(LRSchedule):
    lr: float = 0.001

    def get_lr(self, step: int, total_steps: int) -> float:
        return self.lr

@dataclass
class RewardTransform(ABC):
    @abstractmethod
    def transform(self, raw_rewards: torch.Tensor) -> torch.Tensor:
        """Process raw rewards according to the transform.
        
        Args:
            raw_rewards: The raw rewards to process (torch.Tensor)
            
        Returns:
            Processed rewards (torch.Tensor)
        """

class RewardTransformNone(RewardTransform):
    def transform(self, raw_rewards: torch.Tensor) -> torch.Tensor:
        return raw_rewards

@dataclass
class RewardTransformScale(RewardTransform):
    scale: float = 0.01

    def transform(self, raw_rewards: torch.Tensor) -> torch.Tensor:
        return raw_rewards * self.scale

@dataclass
class RolloutMethod(ABC):
    @abstractmethod
    def collect_rollout_data(
        self, envs: gym.vector.VectorEnv, initial_states: torch.Tensor, agent: AgentProtocol, tracker: SessionTracker
        ) -> tuple[RolloutData, list[np.ndarray]]:
        """Collect rollout data using the given strategy.

        It's important to apply exploration noise while collecting data.
        For example:
        
        ```
        with torch.no_grad():
            action_t, _ = agent.actor.output(state_t)
            action_t = agent.actor.exploration(action_t)
            log_prob, _ = agent.actor.get_log_prob_and_entropy(state_t, action_t)
        ```
        
        Args:
            envs: The environment(s) to roll out in (VecEnv)
            initial_states: Current state(s) to start the rollout from (torch.Tensor)
            agent: The agent to generate actions/values (AgentProtocol)
            tracker: The tracker for episodes, timesteps, etc. (SessionTracker)
        
        Returns:
            rollout_data: Collected RolloutData.
            This function is not expected to scale rewards. Regular rewards should be stored instead.
            next_states: The next state(s) after the rollout (list of ndarray).
        """
        pass

    @property
    @abstractmethod
    def n_envs(self) -> int:
        """Number of parallel environments this strategy uses."""
        pass

@dataclass
class AdvantageMethod(ABC):
    @abstractmethod
    def compute_advantage(
        self, data: RolloutData, next_critic_values: torch.Tensor, gamma: float) -> torch.Tensor:
        """
        Compute advantages from the given rollout data.
        It's expected that rollout_data.returns is already populated.
        """
        pass

# advantages = returns - critic_values.detach()
class AdvantageMethodStandard(AdvantageMethod):
    def compute_advantage(
        self, data: RolloutData, next_critic_values: torch.Tensor, gamma: float) -> torch.Tensor:
        advantages = data.returns - data.critic_values.detach()
        return advantages

@dataclass
class AdvantageTransform(ABC):
    @abstractmethod
    def transform(self, raw_advantages: torch.Tensor) -> torch.Tensor:
        """Process raw advantages according to the transform.
        
        Args:
            raw_advantages: The raw advantages to process (torch.Tensor)
            
        Returns:
            Processed advantages (torch.Tensor)
        """
        pass

class AdvantageTransformNone(AdvantageTransform):
    def transform(self, raw_advantages: torch.Tensor) -> torch.Tensor:
        return raw_advantages

@dataclass
class DataLoadMethod(ABC):
    # returns: (n_envs, n_steps, 1)
    # advantages: (n_envs, n_steps, 1)
    def __call__(self, data: RolloutData) -> Iterator[RolloutData]:
        return self._iter(data)

    @abstractmethod
    def _iter(self, data: RolloutData) -> Iterator[RolloutData]:
        pass

class DataLoadMethodSingle(DataLoadMethod):
    def _iter(self, data: RolloutData) -> Iterator[RolloutData]:
        yield data
    
@dataclass
class GradientTransform(ABC):
    @abstractmethod
    def apply(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        """Apply the gradient transform to the given parameters.
        
        Args:
            parameters: The parameters to apply the gradient transform to (Iterable[torch.nn.Parameter])
            
        Returns:
            None
        """
        pass

class GradientTransformNone(GradientTransform):
    def apply(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        pass