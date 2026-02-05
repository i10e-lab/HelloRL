from collections.abc import Callable
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

HIDDEN_SIZES_DEFAULT = [64, 64]

class ActorProtocol(ABC):
    @abstractmethod
    def output(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Given a state, output an action and its log probability.
        
        Args:
            state: The input state (torch.Tensor)

        Returns:
            action: The action to take (torch.Tensor)
            log_prob: The log probability of the action (torch.Tensor)
        """
        pass

    @abstractmethod
    def get_loss(self, data, critic_value_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) -> torch.Tensor:
        """Compute the actor loss given the data and critic.
        Advantages should be precomputed and stored in data.

        Args:
            data: The rollout data (RolloutData)
            critic_value_func: A function that takes state and action tensors and returns critic values
            (Callable[[torch.Tensor, torch.Tensor], torch.Tensor])
        """
        pass

    @abstractmethod
    def exploration(self, action) -> torch.Tensor:
        """Apply exploration noise to the given action.

        Args:
            action: The action to apply exploration to (torch.Tensor)

        Returns:
            explored_action: The action after applying exploration (torch.Tensor)
        """
        pass

    @abstractmethod
    def get_log_prob_and_entropy(self, state, action) -> tuple[torch.Tensor, torch.Tensor]:
        """Given a state and action, output its log probability and entropy.
        
        Args:
            state: The input state (torch.Tensor)
            action: The input action (torch.Tensor)

        Returns:
            log_prob: The log probability of the action (torch.Tensor)
            entropy: The entropy of the action distribution (torch.Tensor)
        """
        pass

@dataclass
class ActorParams:
    pass

@dataclass
class PolicyObjectiveMethod(ABC):
    @abstractmethod
    def compute_policy_objective(self, ratio: torch.Tensor, advantages: torch.Tensor) -> torch.Tensor:
        """Compute the policy objective given the ratio of new to old policy probabilities and the advantages.
        
        Args:
            ratio: Ratio of new to old policy probabilities (torch.Tensor)
            advantages: Advantages computed from the advantage method (torch.Tensor)
            
        Returns:
            Policy objective (torch.Tensor)
        """
        pass

class PolicyObjectiveMethodStandard(PolicyObjectiveMethod):
    def compute_policy_objective(self, ratio: torch.Tensor, advantages: torch.Tensor) -> torch.Tensor:
        return ratio * advantages

@dataclass
class DistributionalActorParams(ActorParams):
    policy_objective_method: PolicyObjectiveMethod = field(
        default_factory=PolicyObjectiveMethodStandard
    )
    entropy_coef: float = 0.0

class DiscreteActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes=HIDDEN_SIZES_DEFAULT):
        super(DiscreteActorNetwork, self).__init__()

        self.hidden_layers = nn.ModuleList()

        prev_size = state_dim

        for size in hidden_sizes:
            layer = nn.Linear(prev_size, size, dtype=torch.float32)
            self.hidden_layers.append(layer)
            prev_size = size

        self.head = nn.Linear(prev_size, action_dim, dtype=torch.float32)

    def forward(self, state): # logits: (batch_size, action_space)
        x = state

        for layer in self.hidden_layers:
            x = torch.relu(layer(x))

        logits = self.head(x)
        return logits

class DistributionalActor(ActorProtocol, nn.Module):
    def __init__(self, params=DistributionalActorParams()):
        super(DistributionalActor, self).__init__()
        self.params = params

    def get_log_prob_and_entropy(self, state, action):
        pass

    def forward(self, state):
        return self.network(state)

    def get_loss(self, data, critic_value_func):
        old_log_probs = data.log_probs.detach()  # stored at rollout time

        new_log_probs, entropies = self.get_log_prob_and_entropy(data.states, data.actions)
        log_ratio = new_log_probs - old_log_probs
        ratio = torch.exp(log_ratio)

        policy_objective = self.params.policy_objective_method.compute_policy_objective(ratio, data.advantages)
        actor_loss = -(policy_objective.sum())
        actor_loss -= (self.params.entropy_coef * entropies.sum())
        
        return actor_loss
    
    def exploration(self, action) -> torch.Tensor:
        return action

class DiscreteActor(DistributionalActor):
    def __init__(self, state_dim, action_dim, hidden_sizes=HIDDEN_SIZES_DEFAULT, params=DistributionalActorParams()):
        super(DiscreteActor, self).__init__(params=params)

        self.network = DiscreteActorNetwork(state_dim, action_dim, hidden_sizes=hidden_sizes)

    def output(self, state): # action: (batch_size, 1), log_prob: (batch_size, 1)
        actor_logits = self.network.forward(state)
        action_distribution = Categorical(logits=actor_logits)
        action = action_distribution.sample().unsqueeze(-1) # shape: (batch_size, 1)
        log_prob = action_distribution.log_prob(action.squeeze()).unsqueeze(-1) # shape: (batch_size, 1)
        
        return action, log_prob

    def get_log_prob_and_entropy(self, state, action):
        actor_logits = self.network.forward(state)
        action_distribution = Categorical(logits=actor_logits)
        log_prob = action_distribution.log_prob(action.squeeze(-1)).unsqueeze(-1)
        entropy = action_distribution.entropy().unsqueeze(-1)  # (batch_size, 1)
        return log_prob, entropy
    
class ContinuousActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, action_range, hidden_sizes=HIDDEN_SIZES_DEFAULT):
        super(ContinuousActorNetwork, self).__init__()

        self.hidden_layers = nn.ModuleList()

        prev_size = state_dim

        for size in hidden_sizes:
            layer = nn.Linear(prev_size, size, dtype=torch.float32)
            self.hidden_layers.append(layer)
            prev_size = size

        self.head = nn.Linear(prev_size, action_dim, dtype=torch.float32)
        self.action_range = action_range

    def forward(self, state): # logits: (batch_size, action_space)
        x = state

        for layer in self.hidden_layers:
            x = torch.relu(layer(x))

        x = torch.tanh(self.head(x))  # Bounds to -1 to 1

        mins = self.action_range.min(dim=0).values
        maxs = self.action_range.max(dim=0).values
        length = maxs - mins
        x = (x * (length / 2)) + ((mins + maxs) / 2)

        return x
    
class StochasticActor(DistributionalActor):
    def __init__(self, state_dim, action_dim, action_range, hidden_sizes=HIDDEN_SIZES_DEFAULT, params=DistributionalActorParams()):
        super(StochasticActor, self).__init__(params=params)
        self.network = ContinuousActorNetwork(state_dim, action_dim, action_range, hidden_sizes=hidden_sizes)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def output(self, state): # action: (batch_size, 1), log_prob: (batch_size, 1)
        # mean, also referred to as 'mu'
        mean = self.network(state)
        std = torch.exp(self.log_std)  # Exp to make positive
        dist = Normal(mean, std)
        action = dist.sample()  # Shape: (batch_size, action_dim)
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)  # Sum for multi-dim, shape: (batch_size, 1)
        
        return action, log_prob

    def get_log_prob_and_entropy(self, state, action):
        mean = self.network(state)
        std = torch.exp(self.log_std)  # Exp to make positive
        dist = Normal(mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)  # (batch_size, 1)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)  # (batch_size, 1)
        return log_prob, entropy

@dataclass
class DeterministicActorParams(ActorParams):
    exploration_std: float = 0.1
    
class DeterministicActor(ActorProtocol, nn.Module):
    def __init__(self, state_dim, action_dim, action_range, hidden_sizes=HIDDEN_SIZES_DEFAULT, params=DeterministicActorParams()):
        super(DeterministicActor, self).__init__()
        self.network = ContinuousActorNetwork(state_dim, action_dim, action_range, hidden_sizes=hidden_sizes)
        self.params = params

    def forward(self, state): # action: (batch_size, 1)
        return self.network(state)
    
    def output(self, state): # action: (batch_size, 1), log_prob: (batch_size, 1)
        # return zeroes instead of logprobs
        return self.network(state), torch.zeros((state.shape[0], 1), dtype=torch.float32)
    
    def get_loss(self, data, critic_value_func):
        actions_pi = self.network(data.states)
        actor_loss = -(critic_value_func(data.states, actions_pi).mean())
        
        return actor_loss
    
    def exploration(self, action) -> torch.Tensor:
        action = action + (torch.randn_like(action) * self.params.exploration_std)

        mins = self.network.action_range.min(dim=0).values
        maxs = self.network.action_range.max(dim=0).values
        action = torch.clamp(action, mins, maxs)

        return action
    
    def get_log_prob_and_entropy(self, state, action):
        # return zeroes instead of logprobs and entropies
        batch_size = state.shape[0]
        log_prob = torch.zeros((batch_size, 1), dtype=torch.float32)
        entropy = torch.zeros((batch_size, 1), dtype=torch.float32)
        return log_prob, entropy