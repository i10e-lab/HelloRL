import torch

from .foundation import *
from .critics import CriticLossMethod

class CriticLossMethodQ(CriticLossMethod):
    def compute_critic_loss(self, target_action_func, target_critic_value_func, new_values, data, gamma):
        with torch.no_grad():
            # this should be the target actor
            a_next = target_action_func(data.next_states)
            # this should be the target critic
            q_next = target_critic_value_func(data.next_states, a_next)
            y = data.rewards + ((gamma * (1.0 - data.terminateds)) * q_next)
        
        # new values come from the current critic, passed into this function
        critic_loss = torch.nn.MSELoss()(new_values, y)

        return critic_loss