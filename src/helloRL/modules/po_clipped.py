import torch

from .actors import *

class PolicyObjectiveMethodClipped(PolicyObjectiveMethod):
    def __init__(self, clip_range: float):
        self.clip_range = clip_range

    def compute_policy_objective(self, ratio: torch.Tensor, advantages: torch.Tensor) -> torch.Tensor:
        clip_low = (1.0 - self.clip_range)
        clip_high = (1.0 + self.clip_range)
        ratio_clipped = torch.clamp(ratio, clip_low, clip_high)
        surr_1 = ratio * advantages
        surr_2 = ratio_clipped * advantages
        policy_objective = torch.minimum(surr_1, surr_2)
        return policy_objective