import torch

from .foundation import *
from .critics import CriticLossMethod

class CriticLossMethodClipped(CriticLossMethod):
    def __init__(self, clip_range_vf: float):
        self.clip_range_vf = clip_range_vf

    def compute_critic_loss(self, reference_actor, reference_critic, new_values, data, gamma):
        old_values = data.critic_values.detach()

        value_delta = (new_values - old_values)
        value_delta_clipped = torch.clamp(value_delta, -self.clip_range_vf, self.clip_range_vf)
        values_clipped = old_values + value_delta_clipped

        loss_unclipped = (new_values - data.returns).pow(2)
        loss_clipped = (values_clipped - data.returns).pow(2)

        return torch.maximum(loss_unclipped, loss_clipped).mean()
