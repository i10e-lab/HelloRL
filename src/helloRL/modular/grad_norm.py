import torch

from helloRL.modular.foundation import *

class GradientTransformClipNorm(GradientTransform):
    def __init__(self, max_norm: float):
        self.max_norm = max_norm

    def apply(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        torch.nn.utils.clip_grad_norm_(parameters, self.max_norm)