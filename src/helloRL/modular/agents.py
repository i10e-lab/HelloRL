import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from dataclasses import dataclass
from copy import deepcopy

from helloRL.modular.actors import ActorProtocol
from helloRL.modular.critics import CriticProtocol

class AgentProtocol(ABC):
    actor: ActorProtocol
    critics: list[CriticProtocol]

    @abstractmethod
    def get_action(self, state):
        """Get the action from the actor for the given state.

        Args:
            state: The input state (torch.Tensor)
        """
        pass

    def get_critic_value(self, state, action):
        """Get the critic value for the given state and action.

        Args:
            state: The input state (torch.Tensor)
            action: The input action (torch.Tensor)
        """
        pass

    @abstractmethod
    def get_target_action(self, state):
        """Get the target actor's output for the given state.
        A simple implementation could use the main actor if no target networks are used.

        Args:
            state: The input state (torch.Tensor)
        """
        pass

    @abstractmethod
    def get_target_critic_value(self, state, action):
        """Get the target critic's output for the given state and action.
        A simple implementation could use the main critics if no target networks are used.

        Args:
            state: The input state (torch.Tensor)
            action: The input action (torch.Tensor)
        """
        pass

    @abstractmethod
    def update_targets(self):
        """Update the target networks, if applicable."""
        pass

@dataclass
class AgentParams:
    pass

class Agent(AgentProtocol):
    def __init__(self, actor: ActorProtocol, critics: list[CriticProtocol], params: AgentParams=AgentParams()):

        super(Agent, self).__init__()

        self.actor = actor
        self.critics = critics
        self.params = params

    def get_action(self, state):
        action, _ = self.actor.output(state)

        return action
    
    def get_critic_value(self, state, action):
        # Return the minimum value across all critics
        values = [critic.output(state, action) for critic in self.critics]
        return torch.min(torch.stack(values), dim=0).values
    
    def get_target_action(self, state):
        return self.get_action(state)
    
    def get_target_critic_value(self, state, action):
        return self.get_critic_value(state, action)
    
    def update_targets(self):
        # No target networks to update
        pass

@dataclass
class AgentWithTargetsParams(AgentParams):
    tau: float = 0.1

class AgentWithTargets(Agent):
    def __init__(self, actor: ActorProtocol, critics: list[CriticProtocol], params: AgentWithTargetsParams):
        super(AgentWithTargets, self).__init__(actor, critics, params)

        self.target_actor_network = deepcopy(self.actor.network)
        self.target_critic_networks = [deepcopy(critic.network) for critic in self.critics]

    def get_target_action(self, state):
        return self.target_actor_network(state)
    
    def get_target_critic_value(self, state, action):
        # Return the minimum value across all target critics
        values = [target_critic_network(state, action) for target_critic_network in self.target_critic_networks]
        return torch.min(torch.stack(values), dim=0).values
    
    def update_targets(self):
        # Soft update targets
        for target_param, param in zip(self.target_actor_network.parameters(), self.actor.network.parameters()):
            target_param.data.copy_((self.params.tau * param.data) + ((1 - self.params.tau) * target_param.data))

        for i, critic in enumerate(self.critics):
            for target_param, param in zip(self.target_critic_networks[i].parameters(), critic.network.parameters()):
                target_param.data.copy_((self.params.tau * param.data) + ((1 - self.params.tau) * target_param.data))