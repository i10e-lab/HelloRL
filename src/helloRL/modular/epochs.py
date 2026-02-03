import torch
from torch.utils.data import DataLoader, TensorDataset

from helloRL.modular.foundation import *

class DataLoadMethodEpochs(DataLoadMethod):
    def __init__(self, n_epochs: int, mb_size: int):
        self.n_epochs = n_epochs
        self.mb_size = mb_size

    def _iter(self, data: RolloutData) -> Iterator[RolloutData]:
        data = data.flatten()

        # Squeeze dimension 0 so DataLoader can batch along the time dimension
        dataset = TensorDataset(
            data.states.squeeze(0), data.actions.squeeze(0), data.next_states.squeeze(0),
            data.rewards.squeeze(0), data.terminateds.squeeze(0), data.truncateds.squeeze(0),
            data.dones.squeeze(0), data.critic_values.squeeze(0), data.log_probs.squeeze(0),
            data.returns.squeeze(0), data.advantages.squeeze(0))
        for epoch in range(self.n_epochs):
            loader = DataLoader(dataset, batch_size=self.mb_size, shuffle=True)

            for batch in loader:
                b_states, b_actions, b_next_states, b_rewards, b_terminateds, b_truncateds, \
                b_dones, b_critic_values, b_log_probs, b_returns, b_advantages = batch

                b_states = b_states.unsqueeze(0)
                b_actions = b_actions.unsqueeze(0)
                b_next_states = b_next_states.unsqueeze(0)
                b_rewards = b_rewards.unsqueeze(0)
                b_terminateds = b_terminateds.unsqueeze(0)
                b_truncateds = b_truncateds.unsqueeze(0)
                b_dones = b_dones.unsqueeze(0)
                b_critic_values = b_critic_values.unsqueeze(0)
                b_log_probs = b_log_probs.unsqueeze(0)
                b_returns = b_returns.unsqueeze(0)
                b_advantages = b_advantages.unsqueeze(0)

                batch_data = RolloutData(
                    states=b_states,
                    actions=b_actions,
                    next_states=b_next_states,
                    rewards=b_rewards,
                    terminateds=b_terminateds,
                    truncateds=b_truncateds,
                    dones=b_dones,
                    critic_values=b_critic_values,
                    log_probs=b_log_probs,
                    returns=b_returns,
                    advantages=b_advantages
                )

                yield batch_data