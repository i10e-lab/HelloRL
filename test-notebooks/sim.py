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
# For testing the features of utils.sim
#
# sim is a wrapper on Gymnasium which adds a few nice features, like overlays and in-line rendering

# %%
import torch
import gymnasium as gym

from helloRL.utils import sim


# %%
class SimpleActor:
    def output(self, state):
        return torch.randn(2),  # Dummy action


# %%
actor = SimpleActor()

env = gym.make('LunarLander-v3', continuous=True, render_mode='rgb_array')
sim.run_sim_once(env, actor, render_params={})
