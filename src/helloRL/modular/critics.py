import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable

from helloRL.modular.rollout_data import RolloutData

class CriticProtocol(ABC):
    @abstractmethod
    def output(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Given a state and action, output its value estimate.
        The critic does not need to pay attention to the action if not needed, but it's provided.
        
        Args:
            state: The input state (torch.Tensor)
            action: The input action (torch.Tensor)

        Returns:
            value: The estimated value of the state-action pair (torch.Tensor)
        """
        pass

    @abstractmethod
    def get_loss(
        self, target_action_func: Callable[[torch.Tensor], torch.Tensor],
        target_critic_value_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        data: RolloutData, gamma: float) -> torch.Tensor:
        """Compute the critic loss given the data and actor.
        The returns should be precomputed and stored in data.
        
        Args:
            target_action_func: A function to get the target actor's output for a given state
            target_critic_value_func: A function to get the target critic's output for a given state and action
            data: The rollout data (RolloutData)
            gamma: The discount factor (float)
        """
        pass

class CriticNetwork(nn.Module):
    def __init__(self, state_dim, hidden_size=64):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size, dtype=torch.float32)
        self.head = nn.Linear(hidden_size, 1, dtype=torch.float32)

    def forward(self, state, action): # value: (batch_size, 1)
        x = torch.relu(self.fc1(state))
        value = self.head(x)
        return value

@dataclass
class CriticLossMethod(ABC):
    @abstractmethod
    # new_values, batch_data, batch_returns, params.gamma
    def compute_critic_loss(
        self, target_action_func: Callable[[torch.Tensor], torch.Tensor], target_critic_value_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], new_values: torch.Tensor,
        data: RolloutData, gamma: float) -> torch.Tensor:
        pass

class CriticLossMethodStandard(CriticLossMethod):
    def compute_critic_loss(self, target_action_func, target_critic_value_func, new_values, data, gamma):
        return nn.MSELoss()(new_values, data.returns)
    
@dataclass
class CriticParams:
    critic_loss_method: CriticLossMethod = field(
        default_factory=CriticLossMethodStandard
    )

class Critic(CriticProtocol, nn.Module):
    def __init__(self, state_dim, hidden_size=64, params=CriticParams()):
        super(Critic, self).__init__()
        self.network = CriticNetwork(state_dim, hidden_size=hidden_size)
        self.params = params

    def forward(self, state, action): # value: (batch_size, 1)
        return self.network(state, action)
    
    def output(self, state, action): # value: (batch_size, 1)
        critic_value = self.network.forward(state, action)

        return critic_value
    
    def get_loss(self, target_action_func, target_critic_value_func, data, gamma):
        new_values = self.network.forward(data.states, data.actions)
        critic_loss = self.params.critic_loss_method.compute_critic_loss(target_action_func, target_critic_value_func, new_values, data, gamma)
        return critic_loss
    
class QCriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=64):
        super(QCriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_size, dtype=torch.float32)
        self.head = nn.Linear(hidden_size, 1, dtype=torch.float32)

    def forward(self, state, action): # value: (batch_size, 1)
        x = torch.cat([state, action], dim=-1)
        x = torch.relu(self.fc1(x))
        value = self.head(x)
        return value
    
class QCritic(CriticProtocol, nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=64, params=CriticParams()):
        super(QCritic, self).__init__()
        self.network = QCriticNetwork(state_dim, action_dim, hidden_size=hidden_size)
        self.params = params

    def forward(self, state, action): # value: (batch_size, 1)
        return self.network(state, action)

    def output(self, state, action): # value: (batch_size, 1)
        q_value = self.network(state, action)

        return q_value
    
    def get_loss(self, target_action_func, target_critic_value_func, data, gamma):
        new_values = self.network(data.states, data.actions)
        critic_loss = self.params.critic_loss_method.compute_critic_loss(target_action_func, target_critic_value_func, new_values, data, gamma)
        return critic_loss