# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: hellorl
#     language: python
#     name: python3
# ---

# %% [markdown]
# For testing the features of modular.replay

# %%
import torch
from helloRL.modular.foundation import *
from helloRL.modular.replay import ReplayBuffer

# %%
torch.manual_seed(0)

state_dim = 2
action_dim = 1
replay_buffer = ReplayBuffer(capacity=6, state_dim=state_dim, action_dim=action_dim)

n_steps = 100
data = RolloutData(
    states=torch.randn((1, n_steps, state_dim)),
    actions=torch.randn((1, n_steps, action_dim)),
    next_states=torch.randn((1, n_steps, state_dim)),
    rewards=torch.randn((1, n_steps, 1)),
    terminateds=torch.zeros((1, n_steps, 1)),
    truncateds=torch.zeros((1, n_steps, 1)),
    dones=torch.zeros((1, n_steps, 1)),
    critic_values=torch.randn((1, n_steps, 1)),
    log_probs=torch.randn((1, n_steps, 1)),
    returns=torch.randn((1, n_steps, 1)),
    advantages=torch.randn((1, n_steps, 1))
)

cropped_data = data[:8]

replay_buffer.add(cropped_data)
cropped_data.rewards, replay_buffer.rewards, replay_buffer.ptr

# %%
# test numerous envs

torch.manual_seed(0)

state_dim = 2
action_dim = 1
replay_buffer = ReplayBuffer(capacity=8, state_dim=state_dim, action_dim=action_dim)

n_steps = 100
n_envs = 2

data = RolloutData(
    states=torch.randn((n_envs, n_steps, state_dim)),
    actions=torch.randn((n_envs, n_steps, action_dim)),
    next_states=torch.randn((n_envs, n_steps, state_dim)),
    rewards=torch.randn((n_envs, n_steps, 1)),
    terminateds=torch.zeros((n_envs, n_steps, 1)),
    truncateds=torch.zeros((n_envs, n_steps, 1)),
    dones=torch.zeros((n_envs, n_steps, 1)),
    critic_values=torch.randn((n_envs, n_steps, 1)),
    log_probs=torch.randn((n_envs, n_steps, 1)),
    returns=torch.randn((n_envs, n_steps, 1)),
    advantages=torch.randn((n_envs, n_steps, 1))
)

cropped_data = data[:6]
cropped_data.rewards

replay_buffer.add(cropped_data)
cropped_data.rewards, replay_buffer.rewards, replay_buffer.ptr
