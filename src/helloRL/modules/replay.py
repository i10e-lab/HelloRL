import torch

from .rollout_data import ExperienceData
from .foundation import *

class ExperienceBuffer(ExperienceData):
    def __init__(self, capacity: int, state_dim: int, action_dim: int):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0

        self.states = torch.zeros((1, capacity, state_dim), dtype=torch.float32)
        self.actions = torch.zeros((1, capacity, action_dim), dtype=torch.float32)
        self.next_states = torch.zeros((1, capacity, state_dim), dtype=torch.float32)
        self.rewards = torch.zeros((1, capacity, 1), dtype=torch.float32)
        self.terminateds = torch.zeros((1, capacity, 1), dtype=torch.float32)
        self.truncateds = torch.zeros((1, capacity, 1), dtype=torch.float32)
        self.dones = torch.zeros((1, capacity, 1), dtype=torch.float32)

    def add(self, data: ExperienceData):
        data = data.flatten()

        n_data = data.states.shape[1]

        max_n_data = min(n_data, self.capacity - self.ptr)

        self.states[:, self.ptr:self.ptr + max_n_data, :] = data.states[:, :max_n_data, :]
        self.actions[:, self.ptr:self.ptr + max_n_data, :] = data.actions[:, :max_n_data, :]
        self.next_states[:, self.ptr:self.ptr + max_n_data, :] = data.next_states[:, :max_n_data, :]
        self.rewards[:, self.ptr:self.ptr + max_n_data, :] = data.rewards[:, :max_n_data, :]
        self.terminateds[:, self.ptr:self.ptr + max_n_data, :] = data.terminateds[:, :max_n_data, :]
        self.truncateds[:, self.ptr:self.ptr + max_n_data, :] = data.truncateds[:, :max_n_data, :]
        self.dones[:, self.ptr:self.ptr + max_n_data, :] = data.dones[:, :max_n_data, :]

        self.ptr = (self.ptr + max_n_data) % self.capacity
        self.size = min((self.size + max_n_data), self.capacity)

        remaining_n_data = n_data - max_n_data

        if remaining_n_data > 0:
            remaining_data = ExperienceData(
                states=data.states[:, max_n_data:, :],
                actions=data.actions[:, max_n_data:, :],
                next_states=data.next_states[:, max_n_data:, :],
                rewards=data.rewards[:, max_n_data:, :],
                terminateds=data.terminateds[:, max_n_data:, :],
                truncateds=data.truncateds[:, max_n_data:, :],
                dones=data.dones[:, max_n_data:, :]
            )
            self.add(remaining_data)

    def sample(self, batch_size: int) -> ExperienceData:
        idx = torch.randint(0, self.size, (batch_size,))
        return ExperienceData(
            states=self.states[:, idx, :],
            actions=self.actions[:, idx, :],
            next_states=self.next_states[:, idx, :],
            rewards=self.rewards[:, idx, :],
            terminateds=self.terminateds[:, idx, :],
            truncateds=self.truncateds[:, idx, :],
            dones=self.dones[:, idx, :]
        )
    
    def all(self) -> ExperienceData:
        return ExperienceData(
            states=self.states[:, :self.size, :],
            actions=self.actions[:, :self.size, :],
            next_states=self.next_states[:, :self.size, :],
            rewards=self.rewards[:, :self.size, :],
            terminateds=self.terminateds[:, :self.size, :],
            truncateds=self.truncateds[:, :self.size, :],
            dones=self.dones[:, :self.size, :]
        )

class ReplayBuffer(RolloutData):
    def __init__(self, capacity: int, state_dim: int, action_dim: int):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0

        self.states = torch.zeros((1, capacity, state_dim), dtype=torch.float32)
        self.actions = torch.zeros((1, capacity, action_dim), dtype=torch.float32)
        self.next_states = torch.zeros((1, capacity, state_dim), dtype=torch.float32)
        self.rewards = torch.zeros((1, capacity, 1), dtype=torch.float32)
        self.terminateds = torch.zeros((1, capacity, 1), dtype=torch.float32)
        self.truncateds = torch.zeros((1, capacity, 1), dtype=torch.float32)
        self.dones = torch.zeros((1, capacity, 1), dtype=torch.float32)
        self.critic_values = torch.zeros((1, capacity, 1), dtype=torch.float32)
        self.log_probs = torch.zeros((1, capacity, 1), dtype=torch.float32)
        self.returns = torch.zeros((1, capacity, 1), dtype=torch.float32)
        self.advantages = torch.zeros((1, capacity, 1), dtype=torch.float32)

    def add(self, data: RolloutData):
        data = data.flatten()

        n_data = data.states.shape[1]

        max_n_data = min(n_data, self.capacity - self.ptr)

        self.states[:, self.ptr:self.ptr + max_n_data, :] = data.states[:, :max_n_data, :]
        self.actions[:, self.ptr:self.ptr + max_n_data, :] = data.actions[:, :max_n_data, :]
        self.next_states[:, self.ptr:self.ptr + max_n_data, :] = data.next_states[:, :max_n_data, :]
        self.rewards[:, self.ptr:self.ptr + max_n_data, :] = data.rewards[:, :max_n_data, :]
        self.terminateds[:, self.ptr:self.ptr + max_n_data, :] = data.terminateds[:, :max_n_data, :]
        self.truncateds[:, self.ptr:self.ptr + max_n_data, :] = data.truncateds[:, :max_n_data, :]
        self.dones[:, self.ptr:self.ptr + max_n_data, :] = data.dones[:, :max_n_data, :]
        self.critic_values[:, self.ptr:self.ptr + max_n_data, :] = data.critic_values[:, :max_n_data, :]
        self.log_probs[:, self.ptr:self.ptr + max_n_data, :] = data.log_probs[:, :max_n_data, :]
        self.returns[:, self.ptr:self.ptr + max_n_data, :] = data.returns[:, :max_n_data, :]
        self.advantages[:, self.ptr:self.ptr + max_n_data, :] = data.advantages[:, :max_n_data, :]
        self.ptr = (self.ptr + max_n_data) % self.capacity
        self.size = min((self.size + max_n_data), self.capacity)

        remaining_n_data = n_data - max_n_data

        if remaining_n_data > 0:
            remaining_data = RolloutData(
                states=data.states[:, max_n_data:, :],
                actions=data.actions[:, max_n_data:, :],
                next_states=data.next_states[:, max_n_data:, :],
                rewards=data.rewards[:, max_n_data:, :],
                terminateds=data.terminateds[:, max_n_data:, :],
                truncateds=data.truncateds[:, max_n_data:, :],
                dones=data.dones[:, max_n_data:, :],
                critic_values=data.critic_values[:, max_n_data:, :],
                log_probs=data.log_probs[:, max_n_data:, :],
                returns=data.returns[:, max_n_data:, :],
                advantages=data.advantages[:, max_n_data:, :]
            )
            self.add(remaining_data)

    def sample(self, batch_size: int) -> RolloutData:
        idx = torch.randint(0, self.size, (batch_size,))
        return RolloutData(
            states=self.states[:, idx, :],
            actions=self.actions[:, idx, :],
            next_states=self.next_states[:, idx, :],
            rewards=self.rewards[:, idx, :],
            terminateds=self.terminateds[:, idx, :],
            truncateds=self.truncateds[:, idx, :],
            dones=self.dones[:, idx, :],
            critic_values=self.critic_values[:, idx, :],
            log_probs=self.log_probs[:, idx, :],
            returns=self.returns[:, idx, :],
            advantages=self.advantages[:, idx, :]
        )

class DataLoadMethodReplay(DataLoadMethod):
    def __init__(self, capacity, state_dim, action_dim, batch_size: int):
        self.replay_buffer = ReplayBuffer(capacity, state_dim=state_dim, action_dim=action_dim)
        self.batch_size = batch_size

    def _iter(self, data: RolloutData) -> Iterator[RolloutData]:
        # store data in replay buffer, then fetch a random sample of mb_size, yield that
        self.replay_buffer.add(data)

        if self.replay_buffer.size < self.batch_size:
            return  # not enough data yet

        # only 1 batch
        batch_data = self.replay_buffer.sample(self.batch_size)

        yield batch_data